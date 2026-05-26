# Hidden Fragility In Confident Predictions

This application is now executable locally and has a completed diagnostic pilot.

## Pilot Run

The pilot used a 5-epoch ResNet-18 trained on 10k CIFAR-10 examples, then evaluated 2k clean test examples and a small CIFAR-10-C subset. The active route was the penultimate embedding PCA subspace with `pca_dim = 2`.

Artifacts:

- `applications/06_hidden_fragility_cifar_c/pilot_k2/outputs/clean_scores.csv`
- `applications/06_hidden_fragility_cifar_c/pilot_k2/outputs/matched_high_confidence_pairs.csv`
- `applications/06_hidden_fragility_cifar_c/pilot_k2/outputs/corruption_records.csv`
- `applications/06_hidden_fragility_cifar_c/pilot_k2/outputs/corruption_group_summary.csv`
- `applications/06_hidden_fragility_cifar_c/pilot_k2/outputs/matched_fragility_contrasts.csv`
- `applications/06_hidden_fragility_cifar_c/pilot_k2/outputs/predictor_benchmark.csv`

```text
clean scored rows: 2000
matched low/high-rho pairs: 12
corruption records: 288
corruptions: gaussian_noise, brightness, contrast, jpeg_compression
severities: 1, 3, 5
```

## Pilot Predictor Benchmark

```text
any_flip AUROC:
scalar baseline        0.8472
gradient baseline      0.8681
gradient + rho         0.8611
rho only               0.6528

mean true-probability drop:
scalar baseline R2     0.2859
gradient baseline R2   0.3343
gradient + rho R2      0.4560
rho only R2            0.0615
```

## Interpretation

The pilot is not paper evidence: the classifier is undertrained and the matched set is small. It is a useful smoke test. It shows that the pipeline runs end to end, that large PCA route dimensions saturate the 10-class Fisher geometry, and that the currently useful diagnostic target is probability collapse rather than binary flip/error.

The full run should use a better trained checkpoint, `pca_dim` in `{1,2,4}`, all CIFAR-10 test examples, all selected CIFAR-10-C corruptions, and severity levels `1..5`.

## Resumable Full Command

```powershell
python scripts\run_hidden_fragility_cifar_c.py --cifar10-dir data\cifar-10-batches-py --cifar10c-dir data\CIFAR-10-C --checkpoint applications\06_hidden_fragility_cifar_c\models\resnet18_cifar10_full.pt --train-if-missing --epochs 40 --batch-size 256 --pca-dim 2 --confidence-quantile 0.70 --seed 20260531
```

The script saves the checkpoint after every epoch. If interrupted, rerun the same command.
