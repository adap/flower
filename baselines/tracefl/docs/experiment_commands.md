# TraceFL Baseline Experiment Commands

This document collects ready-to-use `flwr run` invocations mirroring the
configurations used in the original [TraceFL](https://github.com/SEED-VT/TraceFL)
repository. The commands rely on Flower's `--run-config` flag, which expects a
space-separated list of `key=value` pairs. Flower parses each pair as TOML, so:

* Quote **every string value** (for example `tracefl.dataset='cifar10'`).
* Leave numeric/boolean values unquoted (for example `num-server-rounds=2`).
* When a string contains commas or braces, keep the surrounding single quotes so
  it is treated as a single value.

> **Tip:** Flower does not accept newline characters inside the configuration
> string. Keep each `--run-config "…"` argument on a single line, or pass
> several `--run-config "…"` arguments to split long overrides across lines.

All commands below assume you run them from the `tracefl-baseline` directory in
a Python environment where the project is installed (`pip install -e .`).

## 1. Localization accuracy on correct predictions

Replicates the clean setting reported for CIFAR-10 in Figure 2/Table 3/Figure 5
of the TraceFL paper.

```bash
flwr run . --run-config "num-server-rounds=2 tracefl.dataset='cifar10' tracefl.model='resnet18' tracefl.num-clients=10 tracefl.dirichlet-alpha=0.5 tracefl.max-per-client-data-size=1000 tracefl.max-server-data-size=500 tracefl.batch-size=32 tracefl.provenance-rounds='1,2'"
```

## 2. Varying data heterogeneity (Dirichlet α sweep)

Use this template to sweep over multiple `tracefl.dirichlet-alpha` values as in
Figure 3. Replace `<ALPHA>` with the desired concentration parameter (for
example `0.1`, `0.3`, `0.7`).

```bash
flwr run . --run-config "num-server-rounds=2 tracefl.dataset='mnist' tracefl.model='resnet18' tracefl.num-clients=10 tracefl.dirichlet-alpha=<ALPHA> tracefl.max-per-client-data-size=1000 tracefl.max-server-data-size=500 tracefl.batch-size=32 tracefl.provenance-rounds='1,2'"
```

## 3. Label-flip fault injection

Recreates the label-flipping experiment behind Table 1/Figure 6. Adjust the
`tracefl.faulty-clients-ids` list to target different clients.

```bash
flwr run . --run-config "num-server-rounds=3 tracefl.dataset='mnist' tracefl.model='resnet18' tracefl.num-clients=10 tracefl.dirichlet-alpha=0.7 tracefl.max-per-client-data-size=1000 tracefl.max-server-data-size=500 tracefl.batch-size=32 tracefl.provenance-rounds='1,2,3' tracefl.faulty-clients-ids='[0]' tracefl.label2flip='{"1":0,"2":0,"3":0,"4":0,"5":0,"6":0,"7":0,"8":0,"9":0}'"
```

## 4. Differential privacy ablation

Matches the DP setting discussed in Figure 4/Table 2 by enabling gradient
clipping and Gaussian noise.

```bash
flwr run . --run-config "num-server-rounds=2 tracefl.dataset='mnist' tracefl.model='resnet18' tracefl.num-clients=10 tracefl.dirichlet-alpha=0.3 tracefl.max-per-client-data-size=1000 tracefl.max-server-data-size=500 tracefl.batch-size=32 tracefl.provenance-rounds='1,2' tracefl.noise-multiplier=0.001 tracefl.clipping-norm=15"
```

If you prefer to format the overrides across multiple shell lines, supply
separate `--run-config` arguments. Flower merges them from left to right:

```bash
flwr run . \
  --run-config "num-server-rounds=2 tracefl.dataset='cifar10' tracefl.model='resnet18'" \
  --run-config "tracefl.num-clients=10 tracefl.dirichlet-alpha=0.5 tracefl.max-per-client-data-size=1000" \
  --run-config "tracefl.max-server-data-size=500 tracefl.batch-size=32 tracefl.provenance-rounds='1,2'"
```

Feel free to adjust the round counts or client counts if you need quicker smoke
tests (for example by setting `num-server-rounds=1` or `tracefl.num-clients=2`).

## Collecting CSVs and plotting results

Every provenance round now appends its metrics to a CSV file inside
`results_csvs/`.  The filename encodes the dataset, model, client count,
Dirichlet α, provenance rounds, and optional DP/fault parameters.  These CSVs can
be turned into publication-style figures by calling:

```bash
python -m scripts.generate_graphs \
  --pattern "prov_dataset-mnist_model-resnet18_clients-10_alpha-0.3_rounds-1-2*.csv" \
  --title "TraceFL Localization Accuracy (MNIST, 10 Clients)"
```

Generated artefacts are written to `graphs/png`, `graphs/pdf`, and
`graphs/tables`.  For convenience the repository also ships executable helper
scripts in `scripts/` (mirroring the original TraceFL `scripts` folder) that run
the relevant experiment and plotting command in one go:

```bash
bash scripts/a_figure_2_table_3_and_figure_5.sh
```

Each script documents the scenario it replicates and can be customised by
editing the environment variables at the top of the file.
