# Third-party notices and implementation boundary

## Tail-GAN paper

This repository implements ideas described in:

Rama Cont, Mihai Cucuringu, Renyuan Xu and Chao Zhang, *Tail-GAN: Learning to Simulate Tail Risk Scenarios*, 2025.

## Supplied Tail-GAN source repository

The user supplied a copy of the authors' `Tail-GAN` repository. Its `LICENSE` file specifies GNU General Public License v3.0.

No source file from that repository is included in RiskGraph–TailGAN–LLMTIME. The extension was written as a clean-room implementation using the paper's published mathematical description and a functional audit of the repository. Names of established methods—Tail-GAN, NeuralSort, WGAN, VaR, ES and eigenportfolio—are retained for scientific clarity.

The supplied repository should remain a separate reference archive. Any future direct code reuse must comply with GPLv3 and should not be silently merged into a differently licensed production system.

## LLMTIME paper

The numeric-language and structured distribution extensions are informed by:

Nate Gruver, Marc Finzi, Shikai Qiu and Andrew Gordon Wilson, *Large Language Models Are Zero-Shot Time Series Forecasters*, NeurIPS 2023, arXiv revision 2024.

## Supplied LLMTIME source repository

The user supplied `llmtime-main`, whose `LICENSE` file is the MIT License and attributes copyright to Nate Gruver, Marc Finzi and Shikai Qiu (2023).

The `riskgraph.llmtime` and `riskgraph.performance_v150` packages are clean-room financial implementations. They do not copy the supplied repository's Python source. The supplied code was audited to understand the published serializer, scaling, sampling and likelihood workflow and to identify engineering limitations of its historical dependency stack. The supplied repository should remain a separate reference archive with its original MIT licence.

## Optional external language models

The Hugging Face adapter does not bundle or automatically select a foundation model. Users must supply a local path or model identifier, review the corresponding model and tokenizer licences, record the exact revision and comply with institutional data-governance requirements. Proprietary financial histories should not be transmitted to an external API without explicit authorization.

## RiskGraph reference study

RiskGraph was developed after studying the separately supplied `Stock-Market-Forecasting-main` repository and its associated AppliedMath manuscript. RiskGraph is a modular reimplementation with causal data handling, probabilistic return targets, chronological folds, tests and saved metadata.

## Performance v1.4 method references

The v1.4 design is informed by the published method descriptions of PatchTST, iTransformer, TimeXer, conformalized quantile regression and adaptive conformal inference. No source code from those research repositories is copied into this package. The implementations are project-specific clean-room modules built on PyTorch, NumPy and SciPy.

## Performance v1.5 method and data references

The v1.5 structured Transformer, factor-scale scenario generator, seed-ensemble gate and block-bootstrap confirmation are project-specific clean-room implementations. They use general ideas from probabilistic forecasting, factor models, ensemble learning and dependent-data bootstrap methods; no external research-repository source is copied.

Public research inputs may be downloaded from:

- Federal Reserve Economic Data (FRED) for macroeconomic and financial-condition series;
- the Kenneth R. French Data Library for official daily factor files;
- Yahoo Finance through `yfinance` for reproducible public-market OHLCV research data.

Those data remain subject to the providers' terms, revisions and redistribution restrictions. The package downloads data at runtime and does not redistribute the downloaded histories. Institution-licensed replacements are recommended for publication or production use.
## Performance v1.6 method references

The v1.6 code is a project-specific clean-room implementation. Its design is informed by published concepts from patch-based and self-supervised time-series representation learning, reversible/instance normalization for distribution shift, Tail-GAN, Wasserstein gradient penalties and spectral normalization. No research-repository source is copied.

Relevant method papers include PatchTST, TS2Vec, Reversible Instance Normalization, Tail-GAN, Improved Training of Wasserstein GANs and Spectral Normalization for Generative Adversarial Networks. Their names are used for scientific attribution only.

The statistical expert pool, causal regime features, multi-episode gate, structured distribution adapter and stable factor-scale generator are original project implementations.

