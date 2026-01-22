import os
from typing import List, Dict, Tuple

from .config_cripto import ENCRYPTION_METHOD, ENCRYPTION_ENABLED

CSV_PATH = None
CSV_INITIALIZED = False
TOTAL_CRYPTO_TIME = 0.0
TOTAL_SERIAL_TIME = 0.0
TOTAL_CRYPTO_TIME_BY_GROUP: Dict[str, float] = {}
TOTAL_SERIAL_TIME_BY_GROUP: Dict[str, float] = {}


def reset_crypto_totals() -> None:
    """Reset accumulated crypto/serialization totals."""
    global TOTAL_CRYPTO_TIME, TOTAL_SERIAL_TIME, TOTAL_CRYPTO_TIME_BY_GROUP, TOTAL_SERIAL_TIME_BY_GROUP
    TOTAL_CRYPTO_TIME = 0.0
    TOTAL_SERIAL_TIME = 0.0
    TOTAL_CRYPTO_TIME_BY_GROUP = {}
    TOTAL_SERIAL_TIME_BY_GROUP = {}


def add_crypto_time(
    crypto_time: float, serial_time: float, group: str | None = None
) -> None:
    """Accumulate crypto and serialization time for summary reporting."""
    global TOTAL_CRYPTO_TIME, TOTAL_SERIAL_TIME
    TOTAL_CRYPTO_TIME += crypto_time
    TOTAL_SERIAL_TIME += serial_time
    group_key = group or "unknown"
    TOTAL_CRYPTO_TIME_BY_GROUP[group_key] = (
        TOTAL_CRYPTO_TIME_BY_GROUP.get(group_key, 0.0) + crypto_time
    )
    TOTAL_SERIAL_TIME_BY_GROUP[group_key] = (
        TOTAL_SERIAL_TIME_BY_GROUP.get(group_key, 0.0) + serial_time
    )


def get_crypto_totals(group: str | None = None) -> tuple[float, float]:
    """Return accumulated crypto and serialization totals."""
    if group is not None:
        return (
            TOTAL_CRYPTO_TIME_BY_GROUP.get(group, 0.0),
            TOTAL_SERIAL_TIME_BY_GROUP.get(group, 0.0),
        )
    return TOTAL_CRYPTO_TIME, TOTAL_SERIAL_TIME


def get_crypto_totals_by_group() -> Dict[str, Tuple[float, float]]:
    """Return accumulated crypto/serialization totals by group."""
    return {
        key: (TOTAL_CRYPTO_TIME_BY_GROUP.get(key, 0.0), TOTAL_SERIAL_TIME_BY_GROUP.get(key, 0.0))
        for key in set(TOTAL_CRYPTO_TIME_BY_GROUP) | set(TOTAL_SERIAL_TIME_BY_GROUP)
    }

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
    total_crypto_fit = 0.0
    total_crypto_eval = 0.0
    total_crypto_other = 0.0

    for summary in ROUND_SUMMARIES:
        round_time = summary["round_time"]
        crypto_time = summary["crypto_time"]
        without_crypto = summary["without_crypto"]
        crypto_cumulative = summary.get("crypto_cumulative", crypto_time)
        crypto_fit = summary.get("crypto_fit")
        crypto_eval = summary.get("crypto_eval")
        crypto_other = summary.get("crypto_other")
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
        if crypto_fit is not None:
            parallel_note_parts.append(f"crypto_fit={crypto_fit:.2f}s")
        if crypto_eval is not None:
            parallel_note_parts.append(f"crypto_eval={crypto_eval:.2f}s")
        if crypto_other is not None:
            parallel_note_parts.append(f"crypto_other={crypto_other:.2f}s")
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
        if crypto_fit is not None:
            total_crypto_fit += crypto_fit
        if crypto_eval is not None:
            total_crypto_eval += crypto_eval
        if crypto_other is not None:
            total_crypto_other += crypto_other

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
    if any(value > 0 for value in (total_crypto_fit, total_crypto_eval, total_crypto_other)):
        lines.append(
            "Totale critto per fase: fit={fit:.2f}s | evaluate={evaluate:.2f}s | other={other:.2f}s".format(
                fit=total_crypto_fit,
                evaluate=total_crypto_eval,
                other=total_crypto_other,
            )
        )
    if total_crypto_cumulative != total_crypto_time:
        lines.append(
            "Totale critto cumulativo: {total_crypto:.2f}s".format(
                total_crypto=total_crypto_cumulative
            )
        )

    return lines
