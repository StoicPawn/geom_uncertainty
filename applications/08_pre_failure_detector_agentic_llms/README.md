# Pre-Failure Detector For Agentic LLMs

## Claim

Before a multi-step agent commits an error, accessible-varentropy may reveal that its confidence is internally inaccessible or misplaced.

## Scope

This application uses local synthetic agent-control traces with explicit action choices, not live browser/tool execution. It is a serious pre-failure diagnostic scaffold for decoder-only LLMs.

## Artifacts

- `outputs/agent_step_records.csv`
- `outputs/predictor_benchmark.csv`
- `outputs/trajectory_summary.csv`
- `outputs/rho_group_summary.csv`
- `outputs/policy_need_summary.csv`
- `outputs/qualitative_pre_failure_examples.csv`
- `outputs/checkpoints/*_raw_records.pt`
- `reports/report.md`

## Reproduce

```powershell
python scripts\run_agentic_pre_failure_detector.py --models Qwen/Qwen2.5-0.5B,microsoft/phi-2 --local-files-only --trust-remote-code --resume --max-tasks 10 --top-m 6 --pca-dim 4 --self-consistency-prompts 4 --seed 20260602
```
