# Flower end-to-end tests

This directory contains folders for different scenarios that need to be tested and validated before a change can be added to Flower.

## Local run

```bash
cd framework/e2e/e2e-bare
uv sync --frozen
uv run ./../test_superlink.sh e2e-bare
```

Refresh E2E lockfiles:

```bash
framework/dev/lock-e2e.sh
```
