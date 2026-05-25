# Hidden Fragility In Confident Predictions

Separate vision-domain application for CIFAR-10 -> CIFAR-10-C.

## Claim

Among equally confident and correct predictions, accessible-varentropy should distinguish robust from brittle examples under label-preserving corruptions.

## Status

Not executed in this environment: CIFAR-10 python batches at C:\Users\bottolonif\projects\paper\data\cifar-10-batches-py; CIFAR-10-C directory at C:\Users\bottolonif\projects\paper\data\CIFAR-10-C

## Reproduce

Place CIFAR-10 python batches under `data/cifar-10-batches-py` and CIFAR-10-C `.npy` files under `data/CIFAR-10-C`, then run:

```powershell
python scripts\run_hidden_fragility_cifar_c.py --cifar10-dir data\cifar-10-batches-py --cifar10c-dir data\CIFAR-10-C --train-if-missing --epochs 40 --pca-dim 64 --confidence-quantile 0.70 --seed 20260531
```

The outputs are written to `outputs/` and summarized in `reports/report.md`.
