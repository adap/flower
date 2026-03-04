# Comcast Anomaly Flower FL

This directory now includes a config-driven Flower FL pipeline for the v2 Comcast anomaly models.

## Run (simulation)

```bash
cd /Users/micahsheller/git/flower/examples/research/comcast-anomaly
python scripts/run_comcast_fl.py --config configs/smoke.yaml
```

## Run (deployment, managed local multi-process)

This mode launches one `flower-superlink` plus `N` `flower-supernode` processes on the same machine, then submits runs with `flwr run`.

```bash
cd /Users/micahsheller/git/flower/examples/research/comcast-anomaly
python scripts/run_comcast_fl.py --config configs/deploy_local_smoke.yaml --mode deployment
```

Key deployment settings in config:

- `deployment.launch_mode: managed_local | external | managed_azure_ssh` (default `managed_local`)
- `deployment.connection_name`: local SuperLink connection name used in temporary `FLWR_HOME`
- `deployment.local_num_supernodes`: defaults to `federation.num_clients` when null
- `deployment.run_timeout_sec`, `deployment.poll_interval_sec`
- `deployment.startup_timeout_sec`, `deployment.shutdown_grace_sec`

## Run (deployment, external SuperLink)

Use this for VM/network deployment while keeping the same app codepath:

```bash
python scripts/run_comcast_fl.py --config /path/to/deployment_external.yaml --mode deployment
```

## Run (deployment, Azure multi-VM with TLS + SuperNode auth)

This mode uses SSH inventory against existing Linux VMs. It launches one secure SuperLink, registers SuperNode public keys, launches SuperNodes across VMs, runs both domains sequentially, then mirrors artifacts back locally.

Example config templates:

- `configs/deploy_azure_smoke.yaml`
- `configs/deploy_azure_medium.yaml`

Run:

```bash
cd /Users/micahsheller/git/flower/examples/research/comcast-anomaly
python scripts/run_comcast_fl.py --config configs/deploy_azure_smoke.yaml --mode deployment
```

Prerequisites:

- VMs already provisioned and reachable via SSH from your operator machine.
- `python3`, `flwr`, `flower-superlink`, and `flower-supernode` installed on all target VMs.
- Absolute local paths configured for:
  - `deployment.tls.ca_cert_local_path`
  - `deployment.tls.server_cert_local_path`
  - `deployment.tls.server_key_local_path`
  - `deployment.supernode_auth.private_key_local_paths[*]`
  - `deployment.supernode_auth.public_key_local_paths[*]`
- `deployment.azure_ssh.total_supernodes == federation.num_clients`
- `sum(vms[*].supernodes_on_vm) == total_supernodes`
- By default, hosts must be private IP literals. To allow public IPs explicitly, set `deployment.azure_ssh.allow_public_ips: true`.

Safety defaults in managed Azure mode:

- SuperLink binds to `deployment.azure_ssh.superlink_bind_host` when set, otherwise `superlink_vm.host` (not `0.0.0.0`).
- SuperNode `clientappio` binds to `127.0.0.1` on each VM.
- `artifacts.run_name` and `deployment.connection_name` are restricted to safe identifier patterns.
- Remote workspace path must be an absolute non-root path without `..`.
- Teardown raises an error if remote processes do not exit cleanly.

Managed Azure runtime outputs:

- `artifacts/fl/<run_name>/deployment_runtime/runtime_state.json`
- `artifacts/fl/<run_name>/deployment_runtime/logs/<vm-name>/...` (mirrored remote logs)

## Live UI (observer mode)

Backend-first live dashboard is available under `comcast_ui` (FastAPI + WebSocket).

Launch UI attached to an existing run artifact root:

```bash
cd /Users/micahsheller/git/flower/examples/research/comcast-anomaly
python scripts/run_comcast_ui.py --run-name deploy_local_smoke --run-root artifacts/fl
```

Open:

- `http://127.0.0.1:8050`

Optional: launch FL run as a child process while UI is running:

```bash
python scripts/run_comcast_ui.py \
  --run-name deploy_local_smoke \
  --run-root artifacts/fl \
  --launch-fl \
  --fl-config configs/deploy_local_smoke.yaml \
  --fl-mode deployment
```

Optional heartbeat-driven SuperNode polling controls:

- `--supernode-poll-sec 2.0`
- `--superlink-connection <name>` (explicit connection, useful for external mode)
- `--flwr-home <path>` (explicit `FLWR_HOME` for connection resolution)

Current panel status in this pass:

- Implemented: `federation_topology`, `round_timeline`, `global_quality_trends`
- Stubbed with contracts: `unknown_gate_monitor`, `client_participation`, `non_iid_map`, `update_divergence`, `edge_constraints`, `signal_gallery`, `confusion_regime_explorer`

Telemetry sources:

- Baseline: artifact file polling (`runtime_state.json`, per-domain `metrics.json`, `summary.json`)
- Optional fine-grained: in-process runtime hooks (`UiHookSink`)
- Topology connectivity: `flwr supernode ls ... --format json` polling (online/offline status)

Future VM/Azure path:

- UI APIs/events are runtime-adapter agnostic; Azure support can reuse the same panels and contracts by swapping data collectors/adapters.

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

Managed-local runtime outputs:

- `artifacts/fl/<run_name>/deployment_runtime/runtime_state.json`
- `artifacts/fl/<run_name>/deployment_runtime/logs/superlink.log`
- `artifacts/fl/<run_name>/deployment_runtime/logs/supernode_<id>.log`

## Failure Triage

If deployment run fails or times out:

1. Check `runtime_state.json` for `teardown.all_exited` and process metadata.
2. Review `superlink.log` for control/fleet startup errors.
3. Review `supernode_<id>.log` for connection or ClientApp startup failures.
4. Verify no port conflicts if startup timeout occurs repeatedly.
