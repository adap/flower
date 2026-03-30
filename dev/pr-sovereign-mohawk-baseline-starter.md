# Sovereign Mohawk Baseline Starter

## Overview

The Sovereign Mohawk baseline starter provides an initial scaffold under `baselines/sovereign_mohawk` inspired by the contribution proposal discussed during issue planning.
It is intended as a contributor-friendly starting point for iterating toward a full "verifiable aggregation" baseline following Flower baseline conventions.

## Included components

The baseline currently includes:

- New baseline package layout and Flower app wiring
- Runnable `ServerApp` + `ClientApp` + dataset/model scaffold
- Baseline README with environment setup, run commands, and smoke-test expectations
- Optional verification hooks (`enable-verification-hooks`) that run lightweight checks on aggregated tensors and produce a `verification_report.json` artifact

## What this enables

The baseline is designed to support:

- A contributor-friendly starting point for iterating toward a full "verifiable aggregation" baseline
- Reproducible baseline setup following Flower baseline conventions
- Early operational evidence via the generated `verification_report.json`

## Verification & testing

The following checks and commands are expected to be available for this baseline:

- Updated files should pass standard editor diagnostics and linters.
- The baseline includes explicit lint/test commands:
  - `./dev/format-baseline.sh sovereign_mohawk`
  - `./dev/test-baseline.sh sovereign_mohawk`

These commands are intended to be run locally to validate formatting, linting, and tests before integrating further changes.

## Licensing and conduct

- Contributions to this baseline are expected to be provided under the Apache License 2.0, consistent with project requirements.
- Contributor behavior should follow the project Code of Conduct (Contributor Covenant) and general expectations for respectful collaboration.

## Planned enhancements

The following areas have been identified for future work building on this baseline:

- Extend verification hooks into richer strategy-level checks and metrics
- Add experiment configuration variants and expected result tables
- Add plotting/reporting utilities and benchmark evidence artifacts

This document is intended to serve as a stable reference for the Sovereign Mohawk baseline starter and may be updated as the baseline evolves.
