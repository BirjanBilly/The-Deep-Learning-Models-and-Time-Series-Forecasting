# RiskGraph: Regime-Gated Generative Models for Financial Forecasting

This project implements a probabilistic framework to forecast SPY returns using an EWMA Student-t benchmark, a self-supervised patch Transformer, a stabilized Tail-GAN, and an adaptive generative objective model (GOM).
The project was developed with Python 3.12 and PyTorch 2.5. GPU processing is applied to accelerate the training of full matrix.
The experiment has trained 27 supervised seed models and drawn nine ensemble decisions.


## Key development results

| Development fold | Formal result | Mean pinball improvement over frozen EWMA |
|---|---|---:|
| Crisis 2020 | EWMA Student-t retained | 0.000% |
| Inflation 2022 | Stress-shrunk GOM | **+0.408%** |
| Inflation 2022 | Stabilized Tail-GAN | **+0.285%** |
| Recent 2024 | Stabilized Tail-GAN | **+0.203%** |

In 2022, the GOM and Tail-GAN have reduced the losses significantly. In 2024, the Tail-GAN performed strongly.

<p align="center">
  <img src="docs/figures/figure_3_final_improvements.png" width="760" alt="Formal improvements by fold and model">
</p>


## Repository layout

```text
configs/    experiment configuration
scripts/    data, training, ensemble evaluation, comparison and verification
src/        reusable Python package
tests/     v1.7 regression and safety tests
 docs/      journal-format report and figures
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

## Research paper

The full descriptions of methodology, equations, pseudocode, figures, gate design, horizon results and risk diagnostics are in:

- [Regime-Gated Generative Models for Probabilistic Financial Forecasting](docs/RiskGraph_Regime_Gated_Financial_Forecasting.pdf)

## Results and limitations

See [RESULTS.md](RESULTS.md) for a concise interpretation of accepted and rejected models. 

## Reproducibility and data

See [REPRODUCIBILITY.md](REPRODUCIBILITY.md). 

## Skills demonstrated

- PyTorch model development and debugging
- generative modelling for financial scenarios
- Transformer-based time-series representation learning
- quantile forecasting and proper scoring rules
- chronological validation and model-selection safeguards
- bootstrap inference and VaR backtesting
- reproducible experiment orchestration and regression testing
