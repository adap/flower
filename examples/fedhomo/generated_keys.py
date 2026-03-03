"""fedhomo: Generate TenSEAL CKKS encryption context and keys."""

import logging
from pathlib import Path

import tenseal as ts

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

KEYS_DIR = Path("keys")
SECRET_CONTEXT_PATH = KEYS_DIR / "secret_context.pkl"
PUBLIC_CONTEXT_PATH = KEYS_DIR / "public_context.pkl"

# CKKS parameters — poly_modulus_degree=4096, scale=2**20
# coeff_mod_bit_sizes must satisfy: first + last >= sum(middle) for security
# [30, 20, 30]: total 80 bits, compatible with degree 4096
POLY_MODULUS_DEGREE = 4096
COEFF_MOD_BIT_SIZES = [30, 20, 30]
GLOBAL_SCALE = 2**20


def generate_keys() -> None:
    """Generate CKKS context, save secret and public contexts to disk."""
    KEYS_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Generating CKKS context (poly_modulus_degree=%d)...", POLY_MODULUS_DEGREE)

    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=COEFF_MOD_BIT_SIZES,
    )
    context.generate_galois_keys()
    context.global_scale = GLOBAL_SCALE

    # Save secret context (includes private key — never share this)
    with open(SECRET_CONTEXT_PATH, "wb") as f:
        f.write(context.serialize(save_secret_key=True))
    log.info("Secret context saved to %s", SECRET_CONTEXT_PATH)

    # Make context public and save (safe to distribute)
    context.make_context_public()
    with open(PUBLIC_CONTEXT_PATH, "wb") as f:
        f.write(context.serialize())
    log.info("Public context saved to %s", PUBLIC_CONTEXT_PATH)

    log.info("Keys generated successfully.")


if __name__ == "__main__":
    generate_keys()
