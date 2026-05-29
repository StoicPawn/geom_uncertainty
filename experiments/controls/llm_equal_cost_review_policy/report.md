# LLM Equal-Cost Review Policy Test

This derived test asks whether transparent rho-based scores decide which LLM cases to send to review at the same review budget as entropy, gradient, Jacobian, and Fisher-output selectors.
Correctness labels are used only for evaluation. Within each held-out fold, every selector reviews the same fraction of cases; labels are not used to rank cases.

## Verdict

Rho is useful as a review feature, especially in combination with entropy/gradient under source shift, but it is not the universal best standalone review score. Fisher-output-energy is the strongest selector in grouped-prompt and leave-one-model-out panels at the tested budgets.

## Best Error-Capture Selector

### grouped_prompt

- cost 0.15: best error capture is `fisher_fixed_review` (capture 0.221140, automatic accuracy 0.374237).
- cost 0.25: best error capture is `fisher_fixed_review` (capture 0.348966, automatic accuracy 0.404395).
- cost 0.35: best error capture is `fisher_fixed_review` (capture 0.442750, automatic accuracy 0.418048).

### leave_one_model_out

- cost 0.15: best error capture is `entropy_fixed_review` (capture 0.199733, automatic accuracy 0.362463).
- cost 0.25: best error capture is `fisher_fixed_review` (capture 0.338904, automatic accuracy 0.393939).
- cost 0.35: best error capture is `entropy_fixed_review` (capture 0.421123, automatic accuracy 0.388340).

### leave_one_source_out

- cost 0.15: best error capture is `gradient_fixed_review` (capture 0.189909, automatic accuracy 0.356778).
- cost 0.25: best error capture is `rho_entropy_gradient_low_access_review` (capture 0.302761, automatic accuracy 0.373867).
- cost 0.35: best error capture is `rho_entropy_gradient_low_access_review` (capture 0.438543, automatic accuracy 0.418667).


## Files

```text
llm_equal_cost_review_fold_records.csv
llm_equal_cost_review_summary.csv
llm_equal_cost_review_contrasts.csv
report.md
```
