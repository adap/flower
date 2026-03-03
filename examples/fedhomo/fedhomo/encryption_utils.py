import logging
import pickle

def load_contexts():
    """Load encryption contexts from files"""
    logging.info("Loading encryption contexts")
    public_context = pickle.load(open("fedhomo/public_context.pkl", "rb"))
    secret_context = pickle.load(open("fedhomo/secret_context.pkl", "rb"))
    return public_context, secret_context