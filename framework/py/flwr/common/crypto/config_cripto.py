"""Configurazione di default per i moduli di crittografia."""

ENCRYPTION_ENABLED = False
ENCRYPTION_METHOD = None

INTEGRITY_ENABLED = True
INTEGRITY_METHOD = "HMAC"

AUTH_ENABLED = False
AUTH_METHOD = None

NET = "resnet18"
TLS = False
ACCURACY = 0.5
NUM_CLIENTS = 1
EVALUATION_SIDE = "server"
