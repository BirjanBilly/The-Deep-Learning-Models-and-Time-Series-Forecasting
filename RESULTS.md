# Formal development results

## Headline outcomes

The common benchmark in every fold is the EWMA Student-t quantile forecast.

| Fold | Family | Final mean pinball | Improvement vs EWMA | Decision |
|---|---|---:|---:|---|
| Crisis 2020 | Patch Transformer | 0.00518217 | 0.000% | fallback |
| Crisis 2020 | Tail-GAN | 0.00518217 | 0.000% | fallback |
| Crisis 2020 | GOM | 0.00518217 | 0.000% | fallback |
| Inflation 2022 | Patch Transformer | 0.00412580 | 0.000% | fallback |
| Inflation 2022 | Tail-GAN | 0.00411403 | **+0.285%** | accepted |
| Inflation 2022 | GOM | 0.00410896 | **+0.408%** | accepted |
| Recent 2024 | Patch Transformer | 0.00204523 | 0.000% | fallback |
| Recent 2024 | Tail-GAN | 0.00204107 | **+0.203%** | accepted |
| Recent 2024 | GOM | 0.00204523 | 0.000% | fallback |

## Interpretation

The strongest evidence is the inflation-2022 fold. Both accepted models had positive tuning and confirmation evidence, positive circular block-bootstrap lower bounds, four non-negative confirmation episodes, equal weights across three seeds, and descriptive common-origin loss-difference intervals entirely above zero.

The recent-2024 Tail-GAN forecast also passed the pre-test promotion rule, but its descriptive test interval included zero. It is therefore weaker evidence than the two 2022 results.

In crisis 2020, the raw generators looked favourable on the later test sample, but the validation chronology did not support promotion. The formal system retained EWMA rather than accepting an ex-post result.

## Risk diagnostics

The accepted forecasts improved average quantile scoring but did not solve tail-exception dependence. The 2022 five-day 5% VaR exception rate remained about 9.3%, with strong rejection of exception independence. This separates the forecasting result from a claim of complete risk-model calibration.
