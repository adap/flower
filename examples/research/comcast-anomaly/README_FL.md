# Comcast Anomaly Flower FL

This directory now includes a config-driven Flower FL pipeline for the v2 Comcast anomaly models.

## Run (simulation)

```bash
cd /Users/micahsheller/git/flower/examples/research/comcast-anomaly
python scripts/run_comcast_fl.py --config configs/smoke.yaml
```

## Run (deployment)

Set `mode: deployment` and configure `deployment.superlink` (and optionally `deployment.federation`) in config, then:

```bash
python scripts/run_comcast_fl.py --config /path/to/deployment_config.yaml
```

## Flower App entrypoints

- ClientApp: `comcast_fl.flower_client_app:app`
- ServerApp: `comcast_fl.flower_server_app:app`

## Artifacts

Per-domain outputs:

- `artifacts/fl/<run_name>/<domain>/metrics.json`
- `artifacts/fl/<run_name>/<domain>/checkpoint_best.pt`
- `artifacts/fl/<run_name>/<domain>/confusion_matrix.npy`
- `artifacts/fl/<run_name>/<domain>/threshold.json`

Cross-domain outputs:

- `artifacts/fl/<run_name>/summary.json`
- `artifacts/fl/<run_name>/comparison.csv`
