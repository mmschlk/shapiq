# Baseline Comparison Summary

This section compares the current shapiq text demos with common text-explanation baselines.

## Shared Setup

- **Model:** Gemma causal LM (`google/gemma-4-E2B-it`)
- **Context demo value function:** `v(S) = log P_Gemma("Paris" | question + selected context chunks)`
- **Jailbreak demo value function:** `v(S) = log P_Gemma("Sure" | selected jailbreak prompt segments)`
- **shapiq index:** `k-SII`, order 2

## Environment Availability

| Library | Installed? | Method | Interaction Support | Notes |
| --- | --- | --- | --- | --- |
| shapiq | yes | ExactComputer + k-SII | yes, any-order interactions | Used in the context and jailbreak demos. |
| shap | no | shap.Explainer + shap.maskers.Text | no, mainly first-order text attributions | Useful de-facto baseline for token/text attributions. |
| captum | no | Captum ShapleyValueSampling | no, mainly feature attribution | Can be applied to token or embedding features in PyTorch models. |
| inseq | no | Inseq sequence attribution | no, sequence attribution rather than Shapley interactions | Most relevant for generation-focused explanations. |

## Current Findings

### shapiq

- The context demo identifies a positive `k-SII` interaction between two supporting chunks.
- The context demo identifies a negative `k-SII` interaction between a supporting chunk and a misleading chunk.
- The jailbreak demo identifies the strongest positive interaction between `ignore safety` and `unsafe request`.
- These are pairwise interaction effects, not only first-order token or segment attributions.

### SHAP / Captum / Inseq

- These packages are not installed in the current environment, so a live head-to-head runtime/agreement experiment has not been executed yet.
- Once installed, the comparison should use the same model, inputs, and target scores where possible.
- The key expected difference is that shapiq directly reports interaction indices such as `k-SII`, while the baselines primarily provide first-order attributions or generation attributions.

## Next Step To Run Live Baselines

Install optional baseline packages in a separate environment if needed:

```powershell
python -m pip install shap captum inseq
```

Then add runtime and attribution-agreement measurements for the same context and jailbreak examples.
