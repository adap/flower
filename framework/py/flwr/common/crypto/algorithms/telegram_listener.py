#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
import time
import subprocess
import os
import tomli
import tomli_w  # pip install tomli tomli-w

# --- CONFIG TELEGRAM ---
BOT_TOKEN = "8440783074:AAGBenk_eeglVRWIIvuNACUBCkhSxVJoAio"
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"
#/home/sarahfalco/IdeaProjects/flowerCrypto/examples/customCriptography
PROJECT_DIR = "/home/ns/sarah_falco/flowerCrypto/examples/customCriptography"
PYPROJECT_FILE = os.path.join(PROJECT_DIR, "pyproject.toml")
VENV_DIR = "/home/sarahfalco/IdeaProjects/flowerCrypto/framework/py/flwr/venv"
VENV_PYTHON = "/home/sarahfalco/IdeaProjects/flowerCrypto/framework/py/flwr/venv/bin/python3"

# --- FUNZIONI TELEGRAM ---
def get_updates(offset=None):
    url = f"{BASE_URL}/getUpdates"
    params = {"timeout": 30, "offset": offset}
    resp = requests.get(url, params=params)
    return resp.json()

def send_message(chat_id, text):
    url = f"{BASE_URL}/sendMessage"
    requests.post(url, data={"chat_id": chat_id, "text": text})

def send_file(chat_id, file_path):
    url = f"{BASE_URL}/sendDocument"
    try:
        with open(file_path, "rb") as f:
            requests.post(url, data={"chat_id": chat_id, "caption": "Log finale"}, files={"document": f})
    except Exception as e:
        send_message(chat_id, f"❌ Errore invio file: {e}")


# --- PARSING E RUN ---
def parse_and_run(command: str, chat_id: int):
    if not command.startswith("/run"):
        return
    # parsing dei parametri
    parts = command.replace("/run", "").strip().split()
    params = {}
    for p in parts:
        if "=" in p:
            k, v = p.split("=")
            try:
                v = float(v) if "." in v else int(v)
            except ValueError:
                pass
            params[k] = v




    send_message(chat_id, f"🚀 Lancio Flower nella cartella {PROJECT_DIR} ...")
    try:
        # usa direttamente Python del venv con il modulo flwr.cli.app
        subprocess.run([
            VENV_PYTHON, "-m", "flwr.cli.app", "run", "."
        ], cwd=PROJECT_DIR, check=True)


        send_message(chat_id, "✅ Training completato!")

        # invia CSV finale (nome dinamico)
        csv_files = [
            f
            for f in os.listdir(PROJECT_DIR)
            if f.startswith("serialization_times") and f.endswith(".csv")
        ]
        if csv_files:
            send_file(chat_id, os.path.join(PROJECT_DIR, csv_files[-1]))
        else:
            send_message(chat_id, "⚠️ CSV finale non trovato")

        # invia anche CSV con metriche client per round
        client_metrics_file = os.path.join(PROJECT_DIR, "logs", "client_metriche_round.csv")
        if os.path.exists(client_metrics_file):
            send_file(chat_id, client_metrics_file)
        else:
            send_message(chat_id, "⚠️ File client_metriche_round.csv non trovato")

    except subprocess.CalledProcessError as e:
        send_message(chat_id, f"❌ Errore nel run: {e}")
    except Exception as e:
        send_message(chat_id, f"❌ Errore imprevisto: {e}")

# --- LOOP PRINCIPALE ---
last_update_id = None
while True:
    updates = get_updates(last_update_id)
    for result in updates.get("result", []):
        last_update_id = result["update_id"] + 1
        msg = result["message"]
        chat_id = msg["chat"]["id"]
        text = msg.get("text", "")
        print("📩 Messaggio:", text)
        parse_and_run(text, chat_id)
    time.sleep(2)
