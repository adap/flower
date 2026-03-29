# PR Draft: Sovereign Mohawk Baseline Starter

## Suggested PR Title

Add sovereign_mohawk baseline starter with optional verification hooks

## Suggested PR Body

## Summary

This PR adds an initial baseline scaffold under `baselines/sovereign_mohawk` inspired by the contribution proposal discussed in issue planning.

Included in this first commit set:

- New baseline package layout and Flower app wiring
- Runnable ServerApp + ClientApp + dataset/model scaffold
- Baseline README with environment setup, run commands, and smoke-test expectations
- Optional verification hooks (`enable-verification-hooks`) that run lightweight checks on aggregated tensors and produce a report artifact

## What this enables

- A contributor-friendly starting point for iterating toward a full "verifiable aggregation" baseline
- Reproducible baseline setup following Flower baseline conventions
- Early operational evidence via `verification_report.json`

## Verification done

- Updated files pass editor diagnostics
- Baseline includes explicit lint/test commands:
  - `./dev/format-baseline.sh sovereign_mohawk`
  - `./dev/test-baseline.sh sovereign_mohawk`

## Contribution Compliance

- License: contribution is provided under Apache License 2.0, consistent with project requirements.
- Code of Conduct: this PR follows the Contributor Covenant expectations for respectful collaboration.
- Review and CI: this PR is intended for standard Flower code-owner review and will only be considered merge-ready after all CI checks pass.

## Merge Readiness Checklist

- [x] Opened as a focused PR against main
- [x] Baseline scaffold is runnable and documented
- [x] Formatting/lint/type checks documented and executed locally
- [ ] All GitHub Actions checks passing
- [ ] Code-owner review completed

## Follow-up scope (next PRs)

- Extend verification hooks into richer strategy-level checks/metrics
- Add experiment configuration variants and expected result tables
- Add plotting/reporting utilities and benchmark evidence artifacts
