# Federated LLM Guardrails Benchmarking

This Flower app benchmarks LLM safety guardrails across regulated domains (healthcare, finance, legal) using federated analytics. Each client runs guardrail evaluations on its own domain-specific test prompts locally and sends only aggregate metrics to the server. The raw prompts never leave the client.

## Fetch the App

Install Flower:
```
pip install flwr
```

Fetch the app:
```
flwr new @aishwarya/federated-guardrails
```

This will create a new directory called federated-guardrails with the following structure:
```
federated-guardrails
├── fedguardrails
│   ├── __init__.py
│   ├── client_app.py   # Defines your ClientApp — runs guardrail checks locally
│   ├── server_app.py   # Defines your ServerApp — aggregates metrics across domains
│   ├── guardrails.py   # Guardrail check implementations (injection, PII, topic)
│   └── datasets.py     # Domain-specific test prompts (healthcare, finance, legal)
├── pyproject.toml       # Project metadata like dependencies and configs
└── README.md
```

## What Gets Federated vs What Stays Local

**Sent to server (metrics only):**
- Accuracy, precision, recall, F1 scores
- Per-guardrail detection counts (injection, PII, topic)
- Average evaluation latency
- Confusion matrix values (TP, TN, FP, FN)

**Stays on client (never shared):**
- The actual test prompts
- Domain-specific context or patient/customer data

## Guardrails Evaluated

- **Prompt Injection**: Jailbreak attempts, instruction override patterns
- **PII Detection**: Requests for SSNs, account numbers, addresses
- **Topic Restriction**: Requests for illegal activities, fraud, evidence tampering

## Run the App

### Run with the Simulation Engine

Install the dependencies defined in pyproject.toml as well as the fedguardrails package.
```
cd federated-guardrails && pip install -e .
```

Run with default settings (3 clients, 1 round, 20 prompts per client):
```
flwr run .
```

You can also override some of the settings for your ClientApp and ServerApp defined in pyproject.toml. For example:
```
flwr run . --run-config "num-server-rounds=3"
```

### Extending the App

**Use production guardrails:** Replace the regex-based checks in `guardrails.py` with real systems like Prediction Guard, NVIDIA NeMo Guardrails, or Llama Guard.

**Add more domains:** Add new entries to the `DOMAINS` dictionary in `datasets.py`.

**Add more guardrails:** Extend `guardrails.py` with additional checks (hallucination detection, bias detection) and update the metrics in `client_app.py`.
