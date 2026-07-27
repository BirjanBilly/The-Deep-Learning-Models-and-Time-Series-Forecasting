# Publishing this portfolio repository on GitHub

This repository is curated for public presentation. It includes source code, configuration, tests, the journal-format PDF, and selected figures. Raw market data, model checkpoints, HPC logs, credentials, and machine-specific paths are excluded.

## Recommended repository metadata

**Repository name**

```text
riskgraph-regime-gated-financial-forecasting
```

**Description**

```text
Regime-gated Tail-GAN, GOM and self-supervised Transformer models for probabilistic financial forecasting, benchmarked against EWMA Student-t forecasts.
```

**Topics**

```text
deep-learning time-series financial-forecasting probabilistic-forecasting generative-adversarial-network transformer pytorch quant-finance risk-management var backtesting reproducible-research
```

## 1. Prepare the local folder on Windows

Extract `RiskGraph_GitHub_Portfolio_Repository.zip`, open PowerShell, and enter the extracted folder:

```powershell
cd "C:\Users\86156\Downloads\RiskGraph_GitHub_Portfolio_Repository"
```

Check that no private artefacts were accidentally added:

```powershell
Get-ChildItem -Recurse -File |
  Where-Object {
    $_.Extension -in '.pt', '.pth', '.ckpt', '.env', '.pem', '.key', '.log' -or
    $_.Length -gt 40MB
  } |
  Select-Object FullName, Length
```

The expected result is empty. Keep raw data, trained checkpoints, evidence bundles, SSH material and full cluster logs outside the public repository.

## 2. Run the local checks

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m compileall -q src scripts tests
python -m ruff check .
python -m pytest tests/test_performance_v170.py -q -p no:cacheprovider
python scripts/run_performance_v170_smoke.py --device cpu
```

The smoke test should end with:

```text
PERFORMANCE V1.7 SMOKE PASSED
```

Remove generated caches and smoke outputs before committing:

```powershell
Remove-Item -Recurse -Force .pytest_cache, .ruff_cache, artifacts -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -Directory -Filter __pycache__ |
  Remove-Item -Recurse -Force
```

## 3. Create and push the repository with GitHub CLI

Install Git and GitHub CLI, authenticate once with `gh auth login`, then run:

```powershell
git init
git add .
git commit -m "Publish verified regime-gated forecasting study"
git branch -M main

gh repo create riskgraph-regime-gated-financial-forecasting `
  --public `
  --source=. `
  --remote=origin `
  --push
```

The `gh repo create --source=. --public --push` workflow creates the GitHub repository, adds the remote and pushes the current branch.

## 4. Alternative: create the empty repository in the browser

On GitHub, select **New repository**, use the recommended name and description, select **Public**, and do not initialize it with a README, `.gitignore`, or license because these files already exist locally. Then run:

```powershell
git init
git add .
git commit -m "Publish verified regime-gated forecasting study"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/riskgraph-regime-gated-financial-forecasting.git
git push -u origin main
```

Replace `YOUR-USERNAME` with the GitHub account name.

## 5. Improve the employer-facing repository page

Open the repository page and edit the **About** panel:

- paste the description above;
- add the listed topics;
- set the website field to the PDF or a personal portfolio page when available;
- use `docs/figures/figure_1_architecture.png` as the social-preview image;
- keep the repository public and pin it to the GitHub profile.

The first screen should show the architecture diagram, the result table, the concise interpretation, the test commands and the link to the journal report without requiring an employer to search through folders.

## 6. Create the verified release

Create a tagged release after the initial push:

```powershell
git tag -a v1.7.1 -m "Verified development study"
git push origin v1.7.1

gh release create v1.7.1 `
  docs/RiskGraph_Regime_Gated_Financial_Forecasting.pdf `
  --title "RiskGraph v1.7.1 — verified development study" `
  --notes "Chronological evaluation of EWMA Student-t, a self-supervised patch Transformer, Tail-GAN and GOM. Accepted development results: GOM +0.408% and Tail-GAN +0.285% in 2022; Tail-GAN +0.203% in 2024. The 2025 holdout is excluded."
```

The release provides a stable employer-facing link to the exact report associated with the tagged source code.

## 7. Suggested profile presentation

Pin the repository to the profile and describe it in a CV or application as:

> Developed a leakage-aware probabilistic forecasting framework in PyTorch that combines an EWMA Student-t benchmark, self-supervised patch Transformers, Tail-GAN/GOM scenario models, causal regime routing, chronological validation, block-bootstrap promotion gates and VaR backtesting. Verified development-fold improvements reached 0.408% in mean pinball loss relative to the frozen EWMA benchmark.

A more conservative interview version is:

> Built and tested a deep-learning research pipeline for financial quantile forecasting. The project emphasizes reproducibility and model-risk controls: flexible models are promoted only when they pass chronological, multi-episode and bootstrap checks; otherwise forecasts revert exactly to EWMA.

## 8. Licensing and provenance

Do not add a permissive software license until code provenance has been reviewed. The repository contains `THIRD_PARTY_NOTICES.md`; it records the literature and external reference implementations consulted during development. If every distributed source file is confirmed to be original or compatibly licensed, add the chosen license in a separate reviewed commit.

## 9. What should remain private

Do not publish:

- raw or institution-licensed market data;
- trained `.pt`, `.pth`, or checkpoint files;
- complete HPC logs or scheduler metadata;
- `.env` files, API keys, SSH keys or hostnames;
- private directory names and usernames;
- the full verified analysis bundle when it contains unnecessary intermediate artefacts.

The public repository should contain enough material to understand, test and discuss the work, while large and restricted artefacts remain outside version control.
