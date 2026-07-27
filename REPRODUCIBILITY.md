# Reproducibility notes

## Public contents

The repository contains source code, configuration, scripts, tests, figures and the journal-format development report.

## Excluded contents

The following are deliberately excluded:

- raw or licensed market data;
- trained `.pt` and `.pth` checkpoints;
- full prediction matrices and large experiment artifacts;
- cluster logs, usernames, hostnames and temporary paths;
- API keys, credentials and environment files;
- the untouched 2025 holdout output.

## Expected workflow

1. Create a Python environment and install the package in editable mode.
2. Obtain the data sources described by the configuration and data scripts.
3. Build the processed panel.
4. Run the smoke test.
5. Run the development matrix with the frozen configuration.
6. Verify exact fallback, quantile monotonicity, gate conditions and common origins.

## Determinism and randomness

The formal development matrix uses seeds 11, 22 and 33. GPU kernels may not be bitwise deterministic across all hardware and software combinations. The formal decision logic is therefore based on exported predictions and auditable gate files rather than an assumption of identical training trajectories on every machine.

## Version

This public package is labelled v1.7.1 because it includes the verified comparison-export and exact-fallback serialization corrections applied after the v1.7.0 training matrix.
