# Hidden Fragility In Confident Predictions

Separate vision-domain application for CIFAR-10 -> CIFAR-10-C.

## Claim

Among equally confident and correct predictions, accessible-varentropy should distinguish robust from brittle examples under label-preserving corruptions.

## Current Status

A partial CPU pilot has been executed under `pilot_k2/` using a 5-epoch ResNet-18 trained on 10k CIFAR-10 examples, 2k clean test examples, PCA route dimension 2, four CIFAR-10-C corruptions, and severities 1/3/5.

This pilot is diagnostic only: the classifier is undertrained and the matched set has only 12 low/high-rho pairs. It does not yet support a paper claim. It is useful because it verifies the full pipeline, exposes the scaling cost, and shows that `pca_dim=64` saturates the 10-class Fisher geometry, while `pca_dim=2` gives a nontrivial route.

## Pilot Artifacts

- `pilot_k2/outputs/clean_scores.csv`
- `pilot_k2/outputs/matched_high_confidence_pairs.csv`
- `pilot_k2/outputs/corruption_records.csv`
- `pilot_k2/outputs/corruption_group_summary.csv`
- `pilot_k2/outputs/matched_fragility_contrasts.csv`
- `pilot_k2/outputs/predictor_benchmark.csv`
- `pilot_k2/reports/report.md`

Pilot summary:

```text
clean scored rows: 2000
matched low/high-rho pairs: 12
corruption records: 288
rho-only AUROC for any flip: 0.6528
gradient baseline AUROC for any flip: 0.8681
gradient+rho AUROC for any flip: 0.8611
gradient+rho R2 for true-probability drop: 0.4560
gradient baseline R2 for true-probability drop: 0.3343
```

Interpretation: in this small pilot, `rho` does not improve flip/error prediction beyond projected-gradient baselines, but it does add signal for probability collapse. A full trained model and larger matched set are needed before making a claim.

## Resumable Full Run

The runner now saves the checkpoint after every epoch. If a run is interrupted, rerun the same command and it will resume from the saved epoch.

```powershell
python scripts\run_hidden_fragility_cifar_c.py --cifar10-dir data\cifar-10-batches-py --cifar10c-dir data\CIFAR-10-C --checkpoint applications\06_hidden_fragility_cifar_c\models\resnet18_cifar10_full.pt --train-if-missing --epochs 40 --batch-size 256 --pca-dim 2 --confidence-quantile 0.70 --seed 20260531
```

Optional checkpoint snapshots after every epoch:

```powershell
python scripts\run_hidden_fragility_cifar_c.py --cifar10-dir data\cifar-10-batches-py --cifar10c-dir data\CIFAR-10-C --checkpoint applications\06_hidden_fragility_cifar_c\models\resnet18_cifar10_full.pt --train-if-missing --keep-epoch-checkpoints --epochs 40 --batch-size 256 --pca-dim 2 --confidence-quantile 0.70 --seed 20260531
```

For a fast diagnostic pass:

```powershell
python scripts\run_hidden_fragility_cifar_c.py --cifar10-dir data\cifar-10-batches-py --cifar10c-dir data\CIFAR-10-C --checkpoint applications\06_hidden_fragility_cifar_c\models\resnet18_cifar10_pilot_5ep_10k.pt --out-dir applications\06_hidden_fragility_cifar_c\pilot_k2 --epochs 5 --batch-size 256 --max-train 10000 --max-test 2000 --pca-dim 2 --confidence-quantile 0.60 --low-entropy-quantile 0.40 --high-margin-quantile 0.60 --match-caliper 2.5 --corruptions gaussian_noise,brightness,contrast,jpeg_compression --severities 1,3,5 --seed 20260531
```
