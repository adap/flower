"""NanoGPT model, Shakespeare data loading, and train/test functions."""

import math
import os
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_DIR = Path(__file__).parent / "data"


def _download_shakespeare() -> str:
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "input.txt"
    if not path.exists():
        text = requests.get(DATA_URL).text
        path.write_text(text)
    return path.read_text()


def _get_meta():
    """Build vocab from the full Shakespeare text (deterministic)."""
    meta_path = DATA_DIR / "meta.pkl"
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            return pickle.load(f)
    text = _download_shakespeare()
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    meta = {"vocab_size": len(chars), "stoi": stoi, "itos": itos}
    with open(meta_path, "wb") as f:
        pickle.dump(meta, f)
    return meta


def _encode(text: str, stoi: dict) -> list:
    return [stoi[c] for c in text]


class ShakespeareDataset(Dataset):
    """Character-level Shakespeare dataset that yields (x, y) pairs."""

    def __init__(self, data: np.ndarray, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return max(1, len(self.data) - self.block_size)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx : idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(
            self.data[idx + 1 : idx + 1 + self.block_size].astype(np.int64)
        )
        return x, y


def load_data(partition_id: int, num_partitions: int, batch_size: int, block_size: int):
    """Load a partition of Shakespeare data for simulation.

    Downloads the full text, splits 90/10 train/val, then partitions the
    training set into ``num_partitions`` contiguous chunks.
    """
    text = _download_shakespeare()
    meta = _get_meta()
    ids = np.array(_encode(text, meta["stoi"]), dtype=np.uint16)

    n = len(ids)
    split = int(n * 0.9)
    train_ids = ids[:split]
    val_ids = ids[split:]

    chunk_size = len(train_ids) // num_partitions
    start = partition_id * chunk_size
    end = start + chunk_size if partition_id < num_partitions - 1 else len(train_ids)
    partition_train = train_ids[start:end]

    train_ds = ShakespeareDataset(partition_train, block_size)
    val_ds = ShakespeareDataset(val_ids, block_size)

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return trainloader, valloader


def load_local_data(data_path: str, batch_size: int, block_size: int):
    """Load Shakespeare data from a local directory for deployment.

    Expects ``data_path`` to contain ``train.bin`` and ``val.bin`` files
    (uint16 numpy arrays of encoded characters).
    """
    data_dir = Path(data_path)
    train_ids = np.fromfile(data_dir / "train.bin", dtype=np.uint16)
    val_ids = np.fromfile(data_dir / "val.bin", dtype=np.uint16)

    train_ds = ShakespeareDataset(train_ids, block_size)
    val_ds = ShakespeareDataset(val_ids, block_size)

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return trainloader, valloader


def load_centralized_data(batch_size: int, block_size: int):
    """Load the full val set as a single dataloader (for server-side eval)."""
    text = _download_shakespeare()
    meta = _get_meta()
    ids = np.array(_encode(text, meta["stoi"]), dtype=np.uint16)
    n = len(ids)
    val_ids = ids[int(n * 0.9) :]
    val_ds = ShakespeareDataset(val_ids, block_size)
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None,
            dropout_p=self.dropout if self.training else 0, is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65  # tiny shakespeare char vocab
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2
    bias: bool = True


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight  # weight tying

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device)
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = (
                idx
                if idx.size(1) <= self.config.block_size
                else idx[:, -self.config.block_size :]
            )
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


def build_model(run_config: dict) -> GPT:
    """Build a GPT model from Flower run config."""
    meta = _get_meta()
    config = GPTConfig(
        block_size=int(run_config["block-size"]),
        vocab_size=meta["vocab_size"],
        n_layer=int(run_config["n-layer"]),
        n_head=int(run_config["n-head"]),
        n_embd=int(run_config["n-embd"]),
        dropout=float(run_config["dropout"]),
    )
    return GPT(config)


# ---------------------------------------------------------------------------
# Train / Test
# ---------------------------------------------------------------------------


def train(model: GPT, trainloader: DataLoader, epochs: int, lr: float, device: str,
          max_steps: int = 0):
    """Train the model and return average loss. max_steps=0 means unlimited."""
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    running_loss = 0.0
    n_batches = 0
    for _ in range(epochs):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
            if max_steps > 0 and n_batches >= max_steps:
                return running_loss / n_batches
    return running_loss / max(n_batches, 1)


def test(model: GPT, testloader: DataLoader, device: str, max_batches: int = 20):
    """Evaluate the model and return (loss, perplexity)."""
    model.to(device)
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            total_loss += loss.item()
            n_batches += 1
            if max_batches > 0 and n_batches >= max_batches:
                break
    avg_loss = total_loss / max(n_batches, 1)
    perplexity = math.exp(avg_loss) if avg_loss < 20 else float("inf")
    return avg_loss, perplexity
