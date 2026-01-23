# setup_config.py

import os

from flwr.common.crypto.algorithms import KOBLITZ

CONFIG_FILE = "config_cripto.py"

# Opzioni disponibili
ENCRYPTION_METHODS = [
    "AES",
    "CHACHA",
    "CHACHA_AEAD",
    "AES_GCM",
]
INTEGRITY_METHODS = ["HMAC"]
AUTH_METHODS = [
    "KOBLITZ_SMALL",
    "KOBLITZ_MEDIUM",
    "KOBLITZ_LARGE",
]
NET_OPTIONS = ["custom_cnn", "resnet18", "resnet34", "tiny_cnn", "squeezenet"]
EVALUATION_OPTIONS = ["server", "client"]

def ask_accuracy(prompt, default=0.5):
    allowed = [0.2, 0.5, 0.7, 0.9]
    allowed_str = [str(a) for a in allowed]
    default_str = str(default)
    while True:
        val = input(f"{prompt} {allowed} (default: {default}): ").strip()
        if val == '':
            return default
        if val in allowed_str:
            return float(val)
        print(f"Valore non valido, scegli tra {allowed}")

def ask_bool(prompt, default=False):
    while True:
        val = input(f"{prompt} [{'y' if default else 'n'}/{'n' if default else 'y'}]: ").strip().lower()
        if val == '':
            return default
        if val in ['y', 'yes']:
            return True
        if val in ['n', 'no']:
            return False
        print("Risposta non valida, usa y/n.")

def ask_string(prompt, default=None):
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val else default

def ask_float(prompt, default=None):
    while True:
        val = input(f"{prompt} [{default}]: ").strip()
        if val == '':
            return default
        try:
            return float(val)
        except ValueError:
            print("Inserisci un numero valido.")

def ask_int(prompt, default=None):
    while True:
        val = input(f"{prompt} [{default}]: ").strip()
        if val == '':
            return default
        try:
            return int(val)
        except ValueError:
            print("Inserisci un numero intero valido.")

def ask_choice(prompt, options, default=None):
    options_str = "/".join(options)
    while True:
        val = input(f"{prompt} [{options_str}] (default: {default}): ").strip()
        if val == '' and default is not None:
            return default
        if val in options:
            return val
        print(f"Valore non valido, scegli tra: {options_str}")

def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

def _write_auth_keys(
    curve_name: str, private_key_path: str, public_key_path: str
) -> None:
    private_key, public_key = KOBLITZ.generate_keypair(curve_name)
    _ensure_parent_dir(private_key_path)
    _ensure_parent_dir(public_key_path)
    with open(private_key_path, "wb") as private_file:
        private_file.write(private_key)
    with open(public_key_path, "wb") as public_file:
        public_file.write(public_key)

# Carica configurazione esistente se presente
def load_existing_config():
    config = {}
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                exec(f.read(), config)
        except Exception as e:
            print(f"Errore nel leggere il file di configurazione: {e}")
    return config

# Configurazione principale
def configure():
    existing = load_existing_config()

    # Crittografia
    ENCRYPTION_ENABLED = ask_bool("Abilitare la crittografia?", existing.get('ENCRYPTION_ENABLED', False))
    ENCRYPTION_METHOD = None
    if ENCRYPTION_ENABLED:
        ENCRYPTION_METHOD = ask_choice("Metodo di crittografia", ENCRYPTION_METHODS,
                                       existing.get('ENCRYPTION_METHOD', "AES"))

    # Integrità
    INTEGRITY_ENABLED = ask_bool("Abilitare il controllo di integrità?", existing.get('INTEGRITY_ENABLED', True))
    INTEGRITY_METHOD = None
    if INTEGRITY_ENABLED:
        INTEGRITY_METHOD = ask_choice("Metodo di integrità", INTEGRITY_METHODS,
                                      existing.get('INTEGRITY_METHOD', "HMAC"))

    # Autenticazione con curve ellittiche
    AUTH_ENABLED = ask_bool(
        "Abilitare autenticazione con curve ellittiche?",
        existing.get("AUTH_ENABLED", False),
    )
    AUTH_METHOD = None
    AUTH_PRIVATE_KEY_PATH = None
    AUTH_PUBLIC_KEY_PATH = None
    if AUTH_ENABLED:
        AUTH_METHOD = ask_choice(
            "Metodo di autenticazione (curve)",
            AUTH_METHODS,
            existing.get("AUTH_METHOD", "KOBLITZ_MEDIUM"),
        )
        AUTH_PRIVATE_KEY_PATH = ask_string(
            "Percorso chiave privata",
            existing.get("AUTH_PRIVATE_KEY_PATH", "auth_private_key.pem"),
        )
        AUTH_PUBLIC_KEY_PATH = ask_string(
            "Percorso chiave pubblica",
            existing.get("AUTH_PUBLIC_KEY_PATH", "auth_public_key.pem"),
        )
        if ask_bool("Generare nuove chiavi di autenticazione?", True):
            _write_auth_keys(
                AUTH_METHOD, AUTH_PRIVATE_KEY_PATH, AUTH_PUBLIC_KEY_PATH
            )

    # Rete e altri parametri
    NET = ask_choice("Tipo di rete", NET_OPTIONS, existing.get('NET', "resnet18"))
    TLS = ask_bool("Attivo TLS?", existing.get('TLS', False))
    ACCURACY = ask_accuracy("Accuratezza iniziale", existing.get('ACCURACY', 0.5))

    # Numero di client
    NUM_CLIENTS = ask_int("Numero di client", existing.get('NUM_CLIENTS', 1))

    EVALUATION_SIDE = ask_choice("Valutazione server-side o client-side?",
                                 EVALUATION_OPTIONS,
                                 existing.get('EVALUATION_SIDE', "server"))

    # Salvataggio config
    with open(CONFIG_FILE, "w") as f:
        f.write(f"ENCRYPTION_ENABLED = {ENCRYPTION_ENABLED}\n")
        f.write(f"ENCRYPTION_METHOD = {repr(ENCRYPTION_METHOD)}\n")
        f.write(f"INTEGRITY_ENABLED = {INTEGRITY_ENABLED}\n")
        f.write(f"INTEGRITY_METHOD = {repr(INTEGRITY_METHOD)}\n")
        f.write(f"AUTH_ENABLED = {AUTH_ENABLED}\n")
        f.write(f"AUTH_METHOD = {repr(AUTH_METHOD)}\n")
        f.write(f"AUTH_PRIVATE_KEY_PATH = {repr(AUTH_PRIVATE_KEY_PATH)}\n")
        f.write(f"AUTH_PUBLIC_KEY_PATH = {repr(AUTH_PUBLIC_KEY_PATH)}\n")
        f.write(f"NET = '{NET}'\n")
        f.write(f"TLS = {TLS}\n")
        f.write(f"ACCURACY = {ACCURACY}\n")
        f.write(f"NUM_CLIENTS = {NUM_CLIENTS}\n")
        f.write(f"EVALUATION_SIDE = '{EVALUATION_SIDE}'\n")

    print(f"\nConfigurazione salvata in {CONFIG_FILE}")


if __name__ == "__main__":
    configure()
