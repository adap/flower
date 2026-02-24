#!/usr/bin/env python3
"""Stima costi CPU e RAM delle primitive crittografiche lato client.

Nota importante:
  Questo script NON viene eseguito automaticamente da Flower/client/server.
  Parte solo se lo lanci esplicitamente da terminale.

Uso:
  python examples/customCriptography/benchmark_crypto_resources.py \
      --clients 5 --sizes-kb 64 256 1024 --iterations 30
"""

from __future__ import annotations

import argparse
import csv
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK_PY = REPO_ROOT / "framework" / "py"
if str(FRAMEWORK_PY) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_PY))

ALGOS_DIR = FRAMEWORK_PY / "flwr" / "common" / "crypto" / "algorithms"
if str(ALGOS_DIR) not in sys.path:
    sys.path.insert(0, str(ALGOS_DIR))

import AES
import AES_GCM
import CHACHA
import CHACHA_AEAD
import HMAC

SUPPORTED_METHODS = ["AES", "AES_GCM", "CHACHA", "CHACHA_AEAD", "HMAC"]

ENCRYPT_FNS = {
    "AES": AES.encrypt,
    "AES_GCM": AES_GCM.encrypt,
    "CHACHA": CHACHA.encrypt,
    "CHACHA_AEAD": CHACHA_AEAD.encrypt,
    "HMAC": HMAC.add_hmac,
}

DECRYPT_FNS = {
    "AES": AES.decrypt,
    "AES_GCM": AES_GCM.decrypt,
    "CHACHA": CHACHA.decrypt,
    "CHACHA_AEAD": CHACHA_AEAD.decrypt,
    "HMAC": HMAC.check_hmac,
}


@dataclass
class BenchResult:
    method: str
    payload_bytes: int
    encrypt_cpu_ms: float
    decrypt_cpu_ms: float
    total_cpu_ms: float
    encrypt_peak_kb: float
    decrypt_peak_kb: float
    total_peak_kb: float
    bytes_out: int
    overhead_bytes: int
    overhead_pct: float


def _measure_peak_kb(fn: Callable[[], bytes]) -> tuple[bytes, float]:
    tracemalloc.start()
    try:
        out = fn()
        _cur, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return out, peak / 1024.0


def benchmark_method(method: str, payload_bytes: int, iterations: int) -> BenchResult:
    payload = b"x" * payload_bytes

    enc_cpu = 0.0
    dec_cpu = 0.0
    enc_peak_total = 0.0
    dec_peak_total = 0.0
    out_size = 0

    for _ in range(iterations):
        start = time.process_time()
        encrypted, enc_peak_kb = _measure_peak_kb(lambda: ENCRYPT_FNS[method](payload))
        enc_cpu += (time.process_time() - start) * 1000.0

        start = time.process_time()
        decrypted, dec_peak_kb = _measure_peak_kb(lambda: DECRYPT_FNS[method](encrypted))
        dec_cpu += (time.process_time() - start) * 1000.0

        if decrypted != payload:
            raise ValueError(f"Decryption mismatch for method={method}")

        enc_peak_total += enc_peak_kb
        dec_peak_total += dec_peak_kb
        out_size = len(encrypted)

    avg_enc_cpu = enc_cpu / iterations
    avg_dec_cpu = dec_cpu / iterations
    avg_enc_peak = enc_peak_total / iterations
    avg_dec_peak = dec_peak_total / iterations

    overhead_bytes = out_size - payload_bytes
    overhead_pct = (overhead_bytes / payload_bytes * 100.0) if payload_bytes else 0.0

    return BenchResult(
        method=method,
        payload_bytes=payload_bytes,
        encrypt_cpu_ms=avg_enc_cpu,
        decrypt_cpu_ms=avg_dec_cpu,
        total_cpu_ms=avg_enc_cpu + avg_dec_cpu,
        encrypt_peak_kb=avg_enc_peak,
        decrypt_peak_kb=avg_dec_peak,
        total_peak_kb=avg_enc_peak + avg_dec_peak,
        bytes_out=out_size,
        overhead_bytes=overhead_bytes,
        overhead_pct=overhead_pct,
    )


def write_csv(results: list[BenchResult], path: Path, clients: int) -> None:
    headers = [
        "method",
        "payload_bytes",
        "encrypt_cpu_ms_per_client",
        "decrypt_cpu_ms_per_client",
        "total_cpu_ms_per_client",
        "total_cpu_ms_all_clients",
        "encrypt_peak_kb_per_client",
        "decrypt_peak_kb_per_client",
        "total_peak_kb_per_client",
        "total_peak_kb_all_clients_theoretical",
        "bytes_out",
        "overhead_bytes",
        "overhead_pct",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for r in results:
            writer.writerow(
                [
                    r.method,
                    r.payload_bytes,
                    f"{r.encrypt_cpu_ms:.6f}",
                    f"{r.decrypt_cpu_ms:.6f}",
                    f"{r.total_cpu_ms:.6f}",
                    f"{r.total_cpu_ms * clients:.6f}",
                    f"{r.encrypt_peak_kb:.6f}",
                    f"{r.decrypt_peak_kb:.6f}",
                    f"{r.total_peak_kb:.6f}",
                    f"{r.total_peak_kb * clients:.6f}",
                    r.bytes_out,
                    r.overhead_bytes,
                    f"{r.overhead_pct:.6f}",
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, default=2, help="Numero client")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=SUPPORTED_METHODS,
        choices=SUPPORTED_METHODS,
        help="Primitive da testare",
    )
    parser.add_argument(
        "--sizes-kb",
        nargs="*",
        type=int,
        default=[64, 256, 1024],
        help="Dimensioni payload in KB",
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("examples/customCriptography/crypto_resource_report.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sizes = [s * 1024 for s in args.sizes_kb]
    results: list[BenchResult] = []

    for size in sizes:
        for method in args.methods:
            results.append(benchmark_method(method, size, args.iterations))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(results, args.output, args.clients)

    print(f"Report scritto in: {args.output}")
    print("\nTop primitive per CPU (meno è meglio):")
    for r in sorted(results, key=lambda x: x.total_cpu_ms)[:5]:
        print(
            f"- {r.method:10s} size={r.payload_bytes/1024:.0f}KB "
            f"cpu={r.total_cpu_ms:.3f}ms/client ram_peak={r.total_peak_kb:.2f}KB/client"
        )


if __name__ == "__main__":
    main()
