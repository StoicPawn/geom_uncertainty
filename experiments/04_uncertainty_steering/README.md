# Experiment 4: Uncertainty Steering

## Purpose

Test whether the accessible component of varentropy can be used to steer uncertainty while preserving the model answer as much as possible.

## Five Checks

1. Directional control: `+delta_acc` increases entropy and `-delta_acc` decreases entropy.
2. Equal Fisher-output-energy controls: compare at matched `||F^{1/2} J delta z||`.
3. Specificity: measure uncertainty movement separately from top-1, correctness, ranking, and semantic proxies.
4. Rho dependency: test whether `rho` predicts movement after scalar and geometric controls.
5. Replication: repeat across local MLMs, tasks, layers, dimensions, seeds, and include decoder evidence.

## Data

- `data/prompt_tables.csv`: factual and template prompt table used by the full steering battery.

## Outputs

- `outputs/directionality_summary.csv`: directional steering rates.
- `outputs/equal_output_energy_contrasts.csv`: matched Fisher-output-energy contrasts.
- `outputs/specificity_summary.csv`: answer-stability and semantic-proxy checks.
- `outputs/rho_dependency.csv`: controlled rho-dependency analysis.
- `outputs/replication_matrix.csv`: model/layer/task/dimension coverage.
- `outputs/subspace_scores.csv`: subspace-level scores.
- `outputs/steering_records.csv`: per-example steering records.
- `outputs/decoder_qwen_steering_summary.csv`: decoder-only steering summary.
- `outputs/decoder_qwen_steering_contrasts.csv`: decoder-only steering contrasts.
- `outputs/answer_preserving_qualitative_examples.csv`: hand-readable cases where uncertainty changes while the answer token and top-10 neighborhood are preserved.
- `outputs/distilbert_pilot_steering_summary.csv`: original pilot summary.
- `outputs/skipped_models.csv`: unavailable model records.

## Figures

- `figures/fig06_steering_vs_controls_ci.svg`
- `figures/fig06_steering_vs_controls_ci.png`
- `figures/fig06_uncertainty_steering_main.svg`
- `figures/fig06_uncertainty_steering_main.png`

## Reports

- `reports/summary.md`
- `reports/executive_summary.md`
- `reports/full_battery_report.md`
- `reports/decoder_qwen_report.md`
- `reports/distilbert_pilot_report.md`
- `reports/answer_preserving_qualitative_examples.md`

## Claim Support

This experiment supports the claim that accessible directions steer uncertainty more efficiently than controls while preserving answer content at small perturbation sizes. The mapping from claim to CSV and figure is also listed in `../../CLAIM_MAP.md`.

Decoder-only evidence is treated as part of this main experiment and is documented in `decoder_main/README.md`.

## Related Scripts

- `scripts/run_uncertainty_steering_full_battery.py`
- `scripts/run_uncertainty_steering.py`
- `scripts/run_decoder_uncertainty_steering.py`

## Reproduce

Use `config/reproduce.json` for exact commands and seeds.
