import math
from collections import OrderedDict, abc as container_abcs

import torch
import torch.nn as nn
import torch.nn.functional as F


class HadaWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w1a, w1b, w2a, w2b, scale=torch.tensor(1)):
        ctx.save_for_backward(w1a, w1b, w2a, w2b, scale)
        diff_weight = ((w1a@w1b)*(w2a@w2b)) * scale
        return diff_weight

    @staticmethod
    def backward(ctx, grad_out):
        (w1a, w1b, w2a, w2b, scale) = ctx.saved_tensors
        grad_out = grad_out * scale
        temp = grad_out*(w2a@w2b)
        grad_w1a = temp @ w1b.T
        grad_w1b = w1a.T @ temp

        temp = grad_out * (w1a@w1b)
        grad_w2a = temp @ w2b.T
        grad_w2b = w2a.T @ temp
        
        del temp
        return grad_w1a, grad_w1b, grad_w2a, grad_w2b, None


class HadaWeightCP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t1, w1a, w1b, t2, w2a, w2b, scale=torch.tensor(1)):
        ctx.save_for_backward(t1, w1a, w1b, t2, w2a, w2b, scale)
        
        rebuild1 = torch.einsum('i j k l, j r, i p -> p r k l', t1, w1b, w1a)
        rebuild2 = torch.einsum('i j k l, j r, i p -> p r k l', t2, w2b, w2a)
        
        return rebuild1*rebuild2*scale

    @staticmethod
    def backward(ctx, grad_out):
        (t1, w1a, w1b, t2, w2a, w2b, scale) = ctx.saved_tensors
        grad_out = grad_out * scale
        
        temp = torch.einsum('i j k l, j r -> i r k l', t2, w2b)
        rebuild = torch.einsum('i j k l, i r -> r j k l', temp, w2a)
        
        grad_w = rebuild*grad_out
        del rebuild
        
        grad_w1a = torch.einsum('r j k l, i j k l -> r i', temp, grad_w)
        grad_temp = torch.einsum('i j k l, i r -> r j k l', grad_w, w1a.T)
        del grad_w, temp
        
        grad_w1b = torch.einsum('i r k l, i j k l -> r j', t1, grad_temp)
        grad_t1 = torch.einsum('i j k l, j r -> i r k l', grad_temp, w1b.T)
        del grad_temp
        
        temp = torch.einsum('i j k l, j r -> i r k l', t1, w1b)
        rebuild = torch.einsum('i j k l, i r -> r j k l', temp, w1a)
        
        grad_w = rebuild*grad_out
        del rebuild
        
        grad_w2a = torch.einsum('r j k l, i j k l -> r i', temp, grad_w)
        grad_temp = torch.einsum('i j k l, i r -> r j k l', grad_w, w2a.T)
        del grad_w, temp
        
        grad_w2b = torch.einsum('i r k l, i j k l -> r j', t2, grad_temp)
        grad_t2 = torch.einsum('i j k l, j r -> i r k l', grad_temp, w2b.T)
        del grad_temp
        return grad_t1, grad_w1a, grad_w1b, grad_t2, grad_w2a, grad_w2b, None


def make_weight(w1a, w1b, w2a, w2b, scale):
    return HadaWeight.apply(w1a, w1b, w2a, w2b, scale)


def make_weight_cp(t1, w1a, w1b, t2, w2a, w2b, scale):
    return HadaWeightCP.apply(t1, w1a, w1b, t2, w2a, w2b, scale)


class LohaModule(nn.Module):
    """
    Hadamard product Implementaion for Low Rank Adaptation
    """

    def __init__(
        self, 
        lora_name, 
        org_module: nn.Module, 
        multiplier=1.0, lora_dim=4, alpha=1, 
        dropout=0., rank_dropout=0., module_dropout=0.,
        use_tucker=False,
        use_scalar=False,
        **kwargs,
    ):
        """ if alpha == 0 or None, alpha is rank (no scaling). """
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim
        self.tucker=False
        
        self.shape = org_module.weight.shape
        if org_module.__class__.__name__ == 'Conv2d':
            in_dim = org_module.in_channels
            k_size = org_module.kernel_size
            out_dim = org_module.out_channels
            self.tucker = use_tucker and k_size!=(1, 1)
            if self.tucker:
                shape = (out_dim, in_dim, *k_size)
            else:
                shape = (out_dim, in_dim*k_size[0]*k_size[1])
            self.op = F.conv2d
            self.extra_args = {
                "stride": org_module.stride,
                "padding": org_module.padding,
                "dilation": org_module.dilation,
                "groups": org_module.groups
            }
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            shape = (out_dim, in_dim)
            self.op = F.linear
            self.extra_args = {}
        
        if self.tucker:
            self.hada_t1 = nn.Parameter(torch.empty(lora_dim, lora_dim, shape[2], shape[3]))
            self.hada_w1_a = nn.Parameter(torch.empty(lora_dim, shape[0])) # out_dim, 1-mode
            self.hada_w1_b = nn.Parameter(torch.empty(lora_dim, shape[1])) # in_dim , 2-mode
            
            self.hada_t2 = nn.Parameter(torch.empty(lora_dim, lora_dim, shape[2], shape[3]))
            self.hada_w2_a = nn.Parameter(torch.empty(lora_dim, shape[0])) # out_dim, 1-mode
            self.hada_w2_b = nn.Parameter(torch.empty(lora_dim, shape[1])) # in_dim , 2-mode
        else:
            self.hada_w1_a = nn.Parameter(torch.empty(shape[0], lora_dim))
            self.hada_w1_b = nn.Parameter(torch.empty(lora_dim, shape[1]))
            
            self.hada_w2_a = nn.Parameter(torch.empty(shape[0], lora_dim))
            self.hada_w2_b = nn.Parameter(torch.empty(lora_dim, shape[1]))
        
        self.dropout = dropout
        if dropout:
            print("[WARN]LoHa/LoKr haven't implemented normal dropout yet.")
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        
        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer('alpha', torch.tensor(alpha)) # 定数として扱える

        if use_scalar:
            self.scalar = nn.Parameter(torch.tensor(0.0))
        else:
            self.scalar = torch.tensor(1.0)
        # Need more experiments on init method
        if self.tucker:
            torch.nn.init.normal_(self.hada_t1, std=0.1)
            torch.nn.init.normal_(self.hada_t2, std=0.1)
        torch.nn.init.normal_(self.hada_w1_b, std=1)
        torch.nn.init.normal_(self.hada_w1_a, std=0.1)
        torch.nn.init.normal_(self.hada_w2_b, std=1)
        if use_scalar:
            torch.nn.init.normal_(self.hada_w2_a, std=0.1)
        else:
            torch.nn.init.constant_(self.hada_w2_a, 0)

        self.multiplier = multiplier
        self.org_module = [org_module] # remove in applying
        self.grad_ckpt = False
        self.register_load_state_dict_post_hook(self.load_weight_hook)
    
    def load_weight_hook(self, *args, **kwargs):
        self.scalar = nn.Parameter(torch.ones_like(self.scalar))

    def apply_to(self):
        self.org_module[0].forward = self.forward

    def get_weight(self, orig_weight=None):
        if self.tucker:
            weight = make_weight_cp(
                self.hada_t1, self.hada_w1_a, self.hada_w1_b,
                self.hada_t1, self.hada_w2_a, self.hada_w2_b,
                scale = torch.tensor(self.scale),
            )
        else:
            weight = make_weight(
                self.hada_w1_a, self.hada_w1_b,
                self.hada_w2_a, self.hada_w2_b,
                scale = torch.tensor(self.scale),
            )
        if orig_weight is not None:
            weight = weight.reshape(orig_weight.shape)
        if self.training and self.rank_dropout:
            drop = torch.rand(weight.size(0)) < self.rank_dropout
            weight *= drop.view(-1, [1]*len(weight.shape[1:])).to(weight.device)
        return weight
    
    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        # TODO: Remove `args` and the parsing logic when BC allows.
        if len(args) > 0:
            if destination is None:
                destination = args[0]
            if len(args) > 1 and prefix == '':
                prefix = args[1]
            if len(args) > 2 and keep_vars is False:
                keep_vars = args[2]
            # DeprecationWarning is ignored by default

        if destination is None:
            destination = OrderedDict()
            destination._metadata = OrderedDict()

        local_metadata = dict(version=self._version)
        if hasattr(destination, "_metadata"):
            destination._metadata[prefix[:-1]] = local_metadata

        destination[f'{prefix}alpha'] = self.alpha
        destination[f'{prefix}hada_w1_a'] = self.hada_w1_a * self.scalar
        destination[f'{prefix}hada_w1_b'] = self.hada_w1_b
        destination[f'{prefix}hada_w2_a'] = self.hada_w2_a
        destination[f'{prefix}hada_w2_b'] = self.hada_w2_b
        if self.tucker:
            destination[f'{prefix}hada_t1'] = self.hada_t1
            destination[f'{prefix}hada_t2'] = self.hada_t2
        return destination

    @torch.no_grad()
    def apply_max_norm(self, max_norm, device=None):
        orig_norm = (self.get_weight()*self.scalar).norm()
        norm = torch.clamp(orig_norm, max_norm/2)
        desired = torch.clamp(norm, max=max_norm)
        ratio = desired.cpu()/norm.cpu()
        
        scaled = ratio.item() != 1.0
        if scaled:
            self.scalar *= ratio
        
        return scaled, orig_norm*ratio

    @torch.enable_grad()
    def forward(self, x):
        if self.module_dropout and self.training:
            if torch.rand(1) < self.module_dropout:
                return self.op(
                    x,
                    self.org_module[0].weight.data,
                    None if self.org_module[0].bias is None else self.org_module[0].bias.data
                )
        weight = (
            self.org_module[0].weight.data.to(x.device, dtype=self.hada_w1_a.dtype)
            + self.get_weight(self.org_module[0].weight.data) * self.scalar * self.multiplier
        )
        bias = None if self.org_module[0].bias is None else self.org_module[0].bias.data
        return self.op(
            x, 
            weight.view(self.shape),
            bias,
            **self.extra_args
        )
