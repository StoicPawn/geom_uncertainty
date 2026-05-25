# Control: Random-Init Versus Pretrained

## Purpose

Test whether accessibility structure reflects learned representational organization rather than only architectural/Jacobian degrees of freedom.

## Protocol

- Compare pretrained `bert-base-uncased` with the same architecture initialized from config.
- Compare pretrained `google/bert_uncased_L-2_H-128_A-2` with the same architecture initialized from config.
- Use the same prompt suite, layers, top-k lens, and subspace dimension.

## Main Reading

Pretrained models have higher mean `rho` and much larger accessible variance than random-init models. Random-init models still have nonzero accessibility, so architecture contributes, but pretrained and random rank structures are weakly correlated or unstable.

## Outputs

- `outputs/random_init_scores.csv`
- `outputs/random_init_summary.csv`
- `outputs/pretrained_vs_random_contrasts.csv`
- `outputs/prompt_tables.csv`

## Reproduce

Use `config/reproduce.json`.
