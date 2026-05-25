# Decoder-only Main Evidence

This folder documents the decoder-only component of Experiment 4. The checked-in decoder outputs remain in the parent `outputs/` and `reports/` folders to avoid duplicating CSVs.

## Model

- `Qwen/Qwen2.5-0.5B`

## Artifacts

- `../outputs/decoder_qwen_steering_summary.csv`
- `../outputs/decoder_qwen_steering_contrasts.csv`
- `../reports/decoder_qwen_report.md`

## Scope

This is a decoder logit-lens steering protocol. It is now treated as part of the main uncertainty-steering experiment, while still clearly marked as a smaller decoder-only replication rather than the full masked-LM battery.
