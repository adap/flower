import os
from .config_cripto import ENCRYPTION_METHOD, ENCRYPTION_ENABLED

CSV_PATH = None
CSV_INITIALIZED = False

def init_csv():
    global CSV_PATH, CSV_INITIALIZED
    if CSV_INITIALIZED:
        return CSV_PATH
    base_name = f"serialization_times_{ENCRYPTION_METHOD}.csv" if ENCRYPTION_ENABLED else "serialization_times_noCritto.csv"
    CSV_PATH = base_name

    # Sovrascrive se esiste già
    with open(CSV_PATH, mode="w", encoding="utf-8") as f:
        header_msg = f"Encryption enabled: {ENCRYPTION_METHOD}" if ENCRYPTION_ENABLED else "Encryption disabled"
        print(header_msg, flush=True)
        f.write(header_msg + "\n")

    CSV_INITIALIZED = True
    return CSV_PATH

def log_time(msg: str, *args) -> None:
    """
    Scrive il messaggio su console e su CSV.
    Gestisce sia f-string già formattate sia placeholder con args.
    """
    csv_path = init_csv()  # crea il file solo alla prima chiamata

    try:
        output = msg % args if args else msg
    except TypeError:
        output = msg

    print(output, flush=True)

    try:
        with open(csv_path, mode="a", encoding="utf-8") as f:
            f.write(output + "\n")
    except Exception as e:
        print(f"[log_time ERROR] Non è stato possibile scrivere su {csv_path}: {e}", flush=True)
