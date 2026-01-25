import os
from typing import List, Dict

from .config_cripto import ENCRYPTION_METHOD, ENCRYPTION_ENABLED

CSV_PATH = None
CSV_INITIALIZED = False
TOTAL_CRYPTO_TIME = 0.0
TOTAL_SERIAL_TIME = 0.0
TOTAL_AUTH_TIME = 0.0
TOTAL_OVERHEAD_BYTES = 0
TOTAL_PLAINTEXT_BYTES = 0
OVERHEAD_BY_METHOD: Dict[str, int] = {}
OVERHEAD_COUNT_BY_METHOD: Dict[str, int] = {}
OVERHEAD_BY_CATEGORY: Dict[str, int] = {}
OVERHEAD_COUNT_BY_CATEGORY: Dict[str, int] = {}


def reset_crypto_totals() -> None:
    """Reset accumulated crypto/serialization totals."""
    global TOTAL_CRYPTO_TIME, TOTAL_SERIAL_TIME, TOTAL_AUTH_TIME
    global TOTAL_OVERHEAD_BYTES, TOTAL_PLAINTEXT_BYTES
    global OVERHEAD_BY_METHOD, OVERHEAD_COUNT_BY_METHOD
    global OVERHEAD_BY_CATEGORY, OVERHEAD_COUNT_BY_CATEGORY
    TOTAL_CRYPTO_TIME = 0.0
    TOTAL_SERIAL_TIME = 0.0
    TOTAL_AUTH_TIME = 0.0
    TOTAL_OVERHEAD_BYTES = 0
    TOTAL_PLAINTEXT_BYTES = 0
    OVERHEAD_BY_METHOD = {}
    OVERHEAD_COUNT_BY_METHOD = {}
    OVERHEAD_BY_CATEGORY = {}
    OVERHEAD_COUNT_BY_CATEGORY = {}


def add_crypto_time(crypto_time: float, serial_time: float) -> None:
    """Accumulate crypto and serialization time for summary reporting."""
    global TOTAL_CRYPTO_TIME, TOTAL_SERIAL_TIME
    TOTAL_CRYPTO_TIME += crypto_time
    TOTAL_SERIAL_TIME += serial_time


def add_auth_time(auth_time: float) -> None:
    """Accumulate authentication time for summary reporting."""
    global TOTAL_AUTH_TIME
    TOTAL_AUTH_TIME += auth_time


def get_crypto_totals() -> tuple[float, float]:
    """Return accumulated crypto and serialization totals."""
    return TOTAL_CRYPTO_TIME, TOTAL_SERIAL_TIME


def get_auth_totals() -> float:
    """Return accumulated authentication totals."""
    return TOTAL_AUTH_TIME


def add_overhead(
    method: str,
    category: str,
    added_bytes: int,
    base_bytes: int,
) -> None:
    """Accumulate overhead bytes per method and total payload size."""
    global TOTAL_OVERHEAD_BYTES, TOTAL_PLAINTEXT_BYTES
    global OVERHEAD_BY_METHOD, OVERHEAD_COUNT_BY_METHOD
    global OVERHEAD_BY_CATEGORY, OVERHEAD_COUNT_BY_CATEGORY
    TOTAL_OVERHEAD_BYTES += added_bytes
    TOTAL_PLAINTEXT_BYTES += base_bytes
    OVERHEAD_BY_METHOD[method] = OVERHEAD_BY_METHOD.get(method, 0) + added_bytes
    OVERHEAD_COUNT_BY_METHOD[method] = OVERHEAD_COUNT_BY_METHOD.get(method, 0) + 1
    OVERHEAD_BY_CATEGORY[category] = OVERHEAD_BY_CATEGORY.get(category, 0) + added_bytes
    OVERHEAD_COUNT_BY_CATEGORY[category] = (
        OVERHEAD_COUNT_BY_CATEGORY.get(category, 0) + 1
    )


def get_overhead_totals() -> tuple[int, int]:
    """Return total overhead and plaintext bytes."""
    return TOTAL_OVERHEAD_BYTES, TOTAL_PLAINTEXT_BYTES


def build_overhead_report() -> List[str]:
    if not OVERHEAD_BY_METHOD:
        return ["Nessun dato di overhead disponibile."]

    lines = []
    total_overhead, total_plaintext = get_overhead_totals()
    total_impact = (
        (total_overhead / total_plaintext * 100.0)
        if total_plaintext > 0
        else 0.0
    )
    lines.append(
        "Overhead totale: {overhead} B su {base} B ({impact:.2f}%)".format(
            overhead=total_overhead,
            base=total_plaintext,
            impact=total_impact,
        )
    )
    for method, added in sorted(OVERHEAD_BY_METHOD.items()):
        count = OVERHEAD_COUNT_BY_METHOD.get(method, 0)
        avg = (added / count) if count > 0 else 0.0
        impact = (added / total_plaintext * 100.0) if total_plaintext > 0 else 0.0
        lines.append(
            "Overhead {method}: {added} B ({impact:.2f}%) | medio={avg:.2f} B/msg".format(
                method=method,
                added=added,
                impact=impact,
                avg=avg,
            )
        )
    for category, added in sorted(OVERHEAD_BY_CATEGORY.items()):
        count = OVERHEAD_COUNT_BY_CATEGORY.get(category, 0)
        avg = (added / count) if count > 0 else 0.0
        lines.append(
            "Overhead medio {category}: {avg:.2f} B/msg".format(
                category=category,
                avg=avg,
            )
        )
    return lines

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
    total_auth_time = 0.0
    total_crypto_cumulative = 0.0
    total_auth_cumulative = 0.0

    for summary in ROUND_SUMMARIES:
        round_time = summary["round_time"]
        crypto_time = summary["crypto_time"]
        without_crypto = summary["without_crypto"]
        auth_time = summary.get("auth_time", 0.0)
        crypto_cumulative = summary.get("crypto_cumulative", crypto_time)
        auth_cumulative = summary.get("auth_cumulative", auth_time)
        parallel_factor = summary.get("parallel_factor")
        parallel_fit = summary.get("parallel_fit")
        parallel_eval = summary.get("parallel_eval")

        impact = (crypto_time / round_time * 100.0) if round_time > 0 else 0.0
        auth_impact = (auth_time / round_time * 100.0) if round_time > 0 else 0.0

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
            "auth={auth_time:.2f}s ({auth_impact:.2f}%) | "
            "crypto_cum={crypto_cumulative:.2f}s | auth_cum={auth_cumulative:.2f}s | "
            "senza_critto={without_crypto:.2f}s{parallel_note}".format(
                round_num=int(summary["round"]),
                round_time=round_time,
                crypto_time=crypto_time,
                impact=impact,
                auth_time=auth_time,
                auth_impact=auth_impact,
                crypto_cumulative=crypto_cumulative,
                auth_cumulative=auth_cumulative,
                without_crypto=without_crypto,
                parallel_note=parallel_note,
            )
        )

        total_round_time += round_time
        total_crypto_time += crypto_time
        total_auth_time += auth_time
        total_crypto_cumulative += crypto_cumulative
        total_auth_cumulative += auth_cumulative

    total_impact = (
        (total_crypto_time / total_round_time * 100.0)
        if total_round_time > 0
        else 0.0
    )
    total_auth_impact = (
        (total_auth_time / total_round_time * 100.0)
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
    lines.append(
        "Totale auth (parallel): {total_auth:.2f}s su {total_round:.2f}s ({impact:.2f}%)".format(
            total_auth=total_auth_time,
            total_round=total_round_time,
            impact=total_auth_impact,
        )
    )
    if total_crypto_cumulative != total_crypto_time:
        lines.append(
            "Totale critto cumulativo: {total_crypto:.2f}s".format(
                total_crypto=total_crypto_cumulative
            )
        )
    if total_auth_cumulative != total_auth_time:
        lines.append(
            "Totale auth cumulativo: {total_auth:.2f}s".format(
                total_auth=total_auth_cumulative
            )
        )

    return lines
