# Full Battery Executive Summary

Run: `results/uncertainty_steering_full_battery_v1`

Models completed locally:

- `distilbert-base-uncased`
- `bert-base-uncased`
- `google/bert_uncased_L-2_H-128_A-2`

RoBERTa was not available in the local Hugging Face cache, so no RoBERTa run was attempted. The separate decoder-only Qwen run remains in `results/decoder_uncertainty_steering_qwen2_5_0_5b_layer23_pca16_n64`.

## 1. Directional Control

On equal Fisher-output-energy accessible steering:

```text
sign      n     directional_rate    monotonic_abs_rate
decrease  2376  0.9857              0.7816
increase  2376  0.9581              0.7631
```

The entropy slope has the expected sign on average:

```text
decrease slope mean = -0.4907
increase slope mean =  0.5289
```

## 2. Equal Fisher-Output-Energy Controls

All rows below compare at identical `||F^{1/2}JB delta z||`.

Against entropy-gradient-orthogonal control:

```text
epsilon  sign      |dH| ratio   |dVar| ratio
0.02     decrease  32.32        5.23
0.02     increase  32.40        5.16
0.05     decrease  12.63        4.41
0.05     increase  12.71        4.37
0.10     decrease   5.88        3.41
0.10     increase   5.97        3.40
```

Against random controls:

```text
epsilon  sign      |dH| ratio   |dVar| ratio
0.02     decrease  1.81         1.66
0.02     increase  1.83         1.67
0.05     decrease  1.79         1.65
0.05     increase  1.83         1.67
0.10     decrease  1.74         1.60
0.10     increase  1.82         1.65
```

## 3. Specificity

Accessible steering changes uncertainty while mostly preserving monitored content:

```text
epsilon  sign      top1_change  target_correct_change  top5_jaccard
0.02     decrease  0.0129       0.0007                 0.9779
0.02     increase  0.0154       0.0000                 0.9742
0.05     decrease  0.0291       0.0009                 0.9471
0.05     increase  0.0399       0.0010                 0.9406
0.10     decrease  0.0554       0.0016                 0.9100
0.10     increase  0.0865       0.0021                 0.8909
```

The cleanest local regime is `epsilon <= 0.05`: uncertainty moves strongly while target correctness changes below 0.11%.

## 4. Rho Dependency After Controls

Partial correlations after controlling for initial entropy, initial varentropy, Fisher trace, gradient norm, logit norm, subspace dimension, Fisher-output energy, model, task, layer, and subspace family:

```text
subset                 outcome              n      partial_corr
all                    |dH|                 19008  0.3837
all                    |dVar|               19008  0.3245
bert-base-uncased      |dH|                  7128  0.3776
bert-base-uncased      |dVar|                7128  0.2405
distilbert-base        |dH|                  7128  0.3570
distilbert-base        |dVar|                7128  0.3571
google tiny BERT       |dH|                  4752  0.4315
google tiny BERT       |dVar|                4752  0.4398
```

This is the strongest evidence that `rho(B)` explains uncertainty steering beyond standard scalar controls.

## 5. Replication

Coverage:

- 3 MLM models.
- 3 task families: factual deep, factual simple, ambiguous open.
- 8 total model layers evaluated.
- subspace dimensions `k = 1, 4, 8`.
- structured state PCA, structured delta PCA, and 2 random subspace reps.
- 19,008 equal-output-energy accessible rows used in rho-dependency tests.

Main limitation:

- RoBERTa was unavailable locally.
- Decoder-only Qwen is covered by a separate run, but not yet merged into this full-battery aggregate.
