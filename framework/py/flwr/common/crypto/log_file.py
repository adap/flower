import os
from typing import List, Dict

from .config_cripto import ENCRYPTION_METHOD, ENCRYPTION_ENABLED

CSV_PATH = None
CSV_INITIALIZED = False
TOTAL_CRYPTO_TIME = 0.0
TOTAL_SERIAL_TIME = 0.0


def reset_crypto_totals() -> None:
    """Reset accumulated crypto/serialization totals."""
    global TOTAL_CRYPTO_TIME, TOTAL_SERIAL_TIME
    TOTAL_CRYPTO_TIME = 0.0
    TOTAL_SERIAL_TIME = 0.0


def add_crypto_time(crypto_time: float, serial_time: float) -> None:
    """Accumulate crypto and serialization time for summary reporting."""
    global TOTAL_CRYPTO_TIME, TOTAL_SERIAL_TIME
    TOTAL_CRYPTO_TIME += crypto_time
    TOTAL_SERIAL_TIME += serial_time


def get_crypto_totals() -> tuple[float, float]:
    """Return accumulated crypto and serialization totals."""
    return TOTAL_CRYPTO_TIME, TOTAL_SERIAL_TIME

ROUND_SUMMARIES: List[Dict[str, float]] = []


def init_csv():
    global CSV_PATH, CSV_INITIALIZED
    if CSV_INITIALIZED:
        return CSV_PATH

    base_name = (
        f"serialization_times_{ENCRYPTION_METHOD}.csv"
        if ENCRYPTION_ENABLED
        else "serialization_times_noCritto.csv"
    )
    CSV_PATH = base_name

    with open(CSV_PATH, mode="w", encoding="utf-8") as f:
        header_msg = (
            f"Encryption enabled: {ENCRYPTION_METHOD}"
            if ENCRYPTION_ENABLED
            else "Encryption disabled"
        )
        print(header_msg, flush=True)
        f.write(header_msg + "\n")

    CSV_INITIALIZED = True
    return CSV_PATH


def log_time(msg: str, *args) -> None:
    csv_path = init_csv()

    if args:
        try:
            output = msg % args
        except (TypeError, ValueError):
            try:
                output = msg.format(*args)
            except Exception:
                output = " ".join([msg, *[str(a) for a in args]])
    else:
        output = msg

    print(output, flush=True)

    try:
        with open(csv_path, mode="a", encoding="utf-8") as f:
            f.write(output + "\n")
    except Exception as e:
        print(f"[log_time ERROR] Scrittura CSV fallita: {e}", flush=True)


def get_round_summaries() -> List[Dict[str, float]]:
    return list(ROUND_SUMMARIES)


def build_round_time_report() -> List[str]:
    if not ROUND_SUMMARIES:
        return ["Nessun dato di round disponibile."]

    lines = []
    total_round_time = 0.0
    total_crypto_time = 0.0
    total_crypto_cumulative = 0.0

    for summary in ROUND_SUMMARIES:
        round_time = summary["round_time"]
        crypto_time = summary["crypto_time"]
        without_crypto = summary["without_crypto"]
        crypto_cumulative = summary.get("crypto_cumulative", crypto_time)
        parallel_factor = summary.get("parallel_factor")
        parallel_fit = summary.get("parallel_fit")
        parallel_eval = summary.get("parallel_eval")

        impact = (crypto_time / round_time * 100.0) if round_time > 0 else 0.0

        parallel_note_parts = []
        if parallel_factor is not None:
            parallel_note_parts.append(f"parallel_factor={parallel_factor:.0f}")
        if parallel_fit is not None:
            parallel_note_parts.append(f"parallel_fit={parallel_fit:.0f}")
        if parallel_eval is not None:
            parallel_note_parts.append(f"parallel_eval={parallel_eval:.0f}")
        parallel_note = f" | {' | '.join(parallel_note_parts)}" if parallel_note_parts else ""

        lines.append(
            "Round {round_num}: totale={round_time:.2f}s | "
            "crypto={crypto_time:.2f}s ({impact:.2f}%) | "
            "crypto_cum={crypto_cumulative:.2f}s | "
            "senza_critto={without_crypto:.2f}s{parallel_note}".format(
                round_num=int(summary["round"]),
                round_time=round_time,
                crypto_time=crypto_time,
                impact=impact,
                crypto_cumulative=crypto_cumulative,
                without_crypto=without_crypto,
                parallel_note=parallel_note,
            )
        )

        total_round_time += round_time
        total_crypto_time += crypto_time
        total_crypto_cumulative += crypto_cumulative

    total_impact = (
        (total_crypto_time / total_round_time * 100.0)
        if total_round_time > 0
        else 0.0
    )

    lines.append(
        "Totale critto (parallel): {total_crypto:.2f}s su {total_round:.2f}s ({impact:.2f}%)".format(
            total_crypto=total_crypto_time,
            total_round=total_round_time,
            impact=total_impact,
        )
    )
    if total_crypto_cumulative != total_crypto_time:
        lines.append(
            "Totale critto cumulativo: {total_crypto:.2f}s".format(
                total_crypto=total_crypto_cumulative
            )
        )

    return lines
