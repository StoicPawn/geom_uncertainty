from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForMaskedLM, BertTokenizer

from accessible_varentropy.bert_mlm import encode_prompt, hidden_states_and_logits, token_id_for_word
from accessible_varentropy.metrics import fisher_scores_for_sample
from accessible_varentropy.mlm_heads import mlm_head_jacobian, mlm_logits_from_hidden


@dataclass(frozen=True)
class FactualCase:
    prompt: str
    target: str
    topic: str
    fact_id: str
    template_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="bert-base-uncased")
    parser.add_argument("--out-dir", type=Path, default=Path("results/factual_error_bert"))
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--random-subspaces", type=int, default=5)
    parser.add_argument("--hidden-random-projections", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--splits", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.35)
    parser.add_argument("--logreg-c", type=float, default=0.1)
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def factual_error_cases() -> list[FactualCase]:
    cases: list[FactualCase] = []

    def add(topic: str, fact_id: str, target: str, templates: list[str]) -> None:
        for idx, template in enumerate(templates):
            cases.append(
                FactualCase(
                    prompt=template,
                    target=target,
                    topic=topic,
                    fact_id=f"{topic}:{fact_id}",
                    template_id=f"t{idx}",
                )
            )

    capital_pairs = [
        ("France", "paris"),
        ("Germany", "berlin"),
        ("Italy", "rome"),
        ("Spain", "madrid"),
        ("Japan", "tokyo"),
        ("China", "beijing"),
        ("Russia", "moscow"),
        ("Egypt", "cairo"),
        ("Greece", "athens"),
        ("Portugal", "lisbon"),
        ("Austria", "vienna"),
        ("Belgium", "brussels"),
        ("Finland", "helsinki"),
        ("Norway", "oslo"),
        ("Sweden", "stockholm"),
        ("Poland", "warsaw"),
        ("Hungary", "budapest"),
        ("Romania", "bucharest"),
        ("Bulgaria", "sofia"),
        ("Serbia", "belgrade"),
        ("Croatia", "zagreb"),
        ("Turkey", "ankara"),
        ("Iran", "tehran"),
        ("Iraq", "baghdad"),
        ("Syria", "damascus"),
        ("Jordan", "amman"),
        ("Lebanon", "beirut"),
        ("Thailand", "bangkok"),
        ("Vietnam", "hanoi"),
        ("Peru", "lima"),
        ("Chile", "santiago"),
        ("Cuba", "havana"),
        ("Jamaica", "kingston"),
        ("Morocco", "rabat"),
        ("Tunisia", "tunis"),
        ("Ghana", "accra"),
        ("Nigeria", "abuja"),
        ("Senegal", "dakar"),
        ("Uganda", "kampala"),
        ("Rwanda", "kigali"),
        ("Nepal", "kathmandu"),
        ("Pakistan", "islamabad"),
    ]
    for country, capital in capital_pairs:
        add(
            "capital",
            country.lower(),
            capital,
            [
                f"The capital of {country} is [MASK].",
                f"{country}'s capital city is [MASK].",
                f"In {country}, the capital city is [MASK].",
            ],
        )

    languages = [
        ("France", "french"),
        ("Germany", "german"),
        ("Italy", "italian"),
        ("Spain", "spanish"),
        ("Portugal", "portuguese"),
        ("Russia", "russian"),
        ("China", "chinese"),
        ("Japan", "japanese"),
        ("Greece", "greek"),
        ("Turkey", "turkish"),
        ("Poland", "polish"),
        ("Hungary", "hungarian"),
        ("Romania", "romanian"),
        ("Bulgaria", "bulgarian"),
        ("Sweden", "swedish"),
        ("Norway", "norwegian"),
    ]
    for country, language in languages:
        add(
            "language",
            country.lower(),
            language,
            [
                f"The language of {country} is [MASK].",
                f"People in {country} often speak [MASK].",
            ],
        )

    continents = [
        ("France", "europe"),
        ("Germany", "europe"),
        ("Spain", "europe"),
        ("Poland", "europe"),
        ("China", "asia"),
        ("Japan", "asia"),
        ("India", "asia"),
        ("Thailand", "asia"),
        ("Egypt", "africa"),
        ("Morocco", "africa"),
        ("Nigeria", "africa"),
        ("Ghana", "africa"),
    ]
    for country, continent in continents:
        add(
            "continent",
            country.lower(),
            continent,
            [
                f"{country} is located in [MASK].",
                f"The continent containing {country} is [MASK].",
            ],
        )

    currencies = [
        ("Japan", "yen"),
        ("China", "yuan"),
        ("Russia", "ruble"),
        ("India", "rupee"),
        ("Turkey", "lira"),
        ("Mexico", "peso"),
        ("Brazil", "real"),
        ("Britain", "pound"),
        ("Germany", "euro"),
        ("France", "euro"),
        ("Italy", "euro"),
        ("Spain", "euro"),
    ]
    for country, currency in currencies:
        add(
            "currency",
            country.lower(),
            currency,
            [
                f"The currency of {country} is the [MASK].",
                f"People in {country} pay with the [MASK].",
            ],
        )

    science_specs = [
        ("water_freezes", "celsius", ["Water freezes at zero degrees [MASK].", "Zero degrees [MASK] is the freezing point of water."]),
        ("largest_planet", "jupiter", ["The largest planet is [MASK].", "[MASK] is the largest planet in the solar system."]),
        ("nearest_star", "sun", ["The nearest star to Earth is the [MASK].", "Earth orbits the [MASK]."]),
        ("red_planet", "mars", ["The red planet is [MASK].", "[MASK] is often called the red planet."]),
        ("breathing_gas", "oxygen", ["Humans breathe [MASK].", "The gas humans need to breathe is [MASK]."]),
        ("plant_process", "photosynthesis", ["Plants use sunlight for [MASK].", "The process by which plants use sunlight is [MASK]."]),
        ("heart", "blood", ["The heart pumps [MASK].", "[MASK] is pumped by the heart."]),
        ("bee_product", "honey", ["Bees make [MASK].", "[MASK] is made by bees."]),
        ("spider_web", "web", ["A spider spins a [MASK].", "Spiders are known for spinning a [MASK]."]),
        ("gold", "metal", ["Gold is a [MASK].", "Iron and gold are both [MASK]."]),
    ]
    for fact_id, target, templates in science_specs:
        add("science", fact_id, target, templates)

    math_specs = [
        ("2+2", "four", ["Two plus two equals [MASK].", "The sum of two and two is [MASK]."]),
        ("5-2", "three", ["Five minus two equals [MASK].", "Subtracting two from five gives [MASK]."]),
        ("10/2", "five", ["Ten divided by two equals [MASK].", "Half of ten is [MASK]."]),
        ("3+4", "seven", ["Three plus four equals [MASK].", "The sum of three and four is [MASK]."]),
        ("6+3", "nine", ["Six plus three equals [MASK].", "The sum of six and three is [MASK]."]),
        ("12/3", "four", ["Twelve divided by three equals [MASK].", "One third of twelve is [MASK]."]),
        ("square_sides", "four", ["A square has [MASK] sides.", "The number of sides in a square is [MASK]."]),
        ("triangle_sides", "three", ["A triangle has [MASK] sides.", "The number of sides in a triangle is [MASK]."]),
    ]
    for fact_id, target, templates in math_specs:
        add("math", fact_id, target, templates)

    lexical_specs = [
        ("hot", "cold"),
        ("up", "down"),
        ("left", "right"),
        ("day", "night"),
        ("yes", "no"),
        ("black", "white"),
        ("old", "new"),
        ("fast", "slow"),
        ("happy", "sad"),
        ("early", "late"),
    ]
    for word, target in lexical_specs:
        add(
            "antonym",
            word,
            target,
            [
                f"The opposite of {word} is [MASK].",
                f"{word.capitalize()} has the opposite meaning of [MASK].",
            ],
        )

    morphology_specs = [
        ("child", "children"),
        ("mouse", "mice"),
        ("foot", "feet"),
        ("tooth", "teeth"),
        ("goose", "geese"),
        ("man", "men"),
        ("woman", "women"),
        ("person", "people"),
        ("leaf", "leaves"),
        ("knife", "knives"),
    ]
    for singular, plural in morphology_specs:
        add(
            "plural",
            singular,
            plural,
            [
                f"The plural of {singular} is [MASK].",
                f"More than one {singular} are called [MASK].",
            ],
        )

    role_specs = [
        ("teaches", "teacher"),
        ("drives a bus", "driver"),
        ("writes books", "author"),
        ("studies science", "scientist"),
        ("cooks food", "chef"),
        ("flies planes", "pilot"),
        ("paints pictures", "artist"),
        ("plays music", "musician"),
        ("treats patients", "doctor"),
        ("acts in films", "actor"),
    ]
    for action, target in role_specs:
        fact_id = action.replace(" ", "_")
        add(
            "role",
            fact_id,
            target,
            [
                f"A person who {action} is a [MASK].",
                f"Someone who {action} can be called a [MASK].",
            ],
        )

    animal_specs = [
        ("salmon", "fish"),
        ("trout", "fish"),
        ("sparrow", "bird"),
        ("eagle", "bird"),
        ("snake", "reptile"),
        ("lizard", "reptile"),
        ("frog", "amphibian"),
        ("toad", "amphibian"),
        ("shark", "fish"),
        ("whale", "mammal"),
        ("dolphin", "mammal"),
        ("spider", "insect"),
    ]
    for animal, kind in animal_specs:
        add(
            "animal_class",
            animal,
            kind,
            [
                f"A {animal} is a [MASK].",
                f"The {animal} is classified as a [MASK].",
            ],
        )

    culture_specs = [
        ("hamlet", "shakespeare", ["Hamlet was written by [MASK].", "The author of Hamlet was [MASK]."]),
        ("mona_lisa", "leonardo", ["The Mona Lisa was painted by [MASK].", "[MASK] painted the Mona Lisa."]),
        ("relativity", "einstein", ["The theory of relativity is associated with [MASK].", "[MASK] is famous for relativity."]),
        ("gravity", "newton", ["Isaac [MASK] described gravity.", "Gravity is famously linked to Newton and the apple; his surname was [MASK]."]),
        ("romeo", "juliet", ["Romeo and [MASK] are Shakespeare characters.", "The title pair is Romeo and [MASK]."]),
        ("sherlock", "holmes", ["Sherlock [MASK] is a fictional detective.", "The detective Sherlock has surname [MASK]."]),
        ("odyssey", "homer", ["The Odyssey is attributed to [MASK].", "[MASK] is traditionally credited with the Odyssey."]),
        ("darwin", "evolution", ["Charles Darwin is associated with [MASK].", "Darwin is famous for the theory of [MASK]."]),
    ]
    for fact_id, target, templates in culture_specs:
        add("culture", fact_id, target, templates)

    common_specs = [
        ("grass", "green", ["Grass is usually [MASK].", "The usual color of grass is [MASK]."]),
        ("sky", "blue", ["The sky is often [MASK].", "The usual color of a clear sky is [MASK]."]),
        ("coffee", "cup", ["People drink coffee from a [MASK].", "A common container for coffee is a [MASK]."]),
        ("sleep", "bed", ["People usually sleep in a [MASK].", "A person sleeps in a [MASK]."]),
        ("fish_swim", "water", ["Fish usually swim in [MASK].", "[MASK] is where fish swim."]),
        ("bread", "flour", ["Bread is made from [MASK].", "[MASK] is a main ingredient in bread."]),
        ("milk", "cow", ["Milk often comes from a [MASK].", "A [MASK] produces milk."]),
        ("rose", "flower", ["A rose is a [MASK].", "The rose is a kind of [MASK]."]),
    ]
    for fact_id, target, templates in common_specs:
        add("common", fact_id, target, templates)

    seen: set[str] = set()
    unique: list[FactualCase] = []
    for case in cases:
        if case.prompt not in seen:
            seen.add(case.prompt)
            unique.append(case)
    return unique


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < eps:
        return np.zeros_like(v)
    return v / norm


def orthonormalize(matrix: np.ndarray, k: int) -> np.ndarray:
    q, _r = np.linalg.qr(matrix)
    return q[:, :k]


def topk_distribution(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argpartition(logits, -k)[-k:]
    idx = idx[np.argsort(logits[idx])[::-1]]
    values = logits[idx]
    exp = np.exp(values - values.max())
    return idx.astype(np.int64), (exp / exp.sum()).astype(np.float64)


def collect_records(tokenizer, model, cases: list[FactualCase], top_k: int) -> tuple[pd.DataFrame, list[dict]]:
    prompt_rows: list[dict] = []
    records: list[dict] = []
    for prompt_id, case in enumerate(cases):
        target_id = token_id_for_word(tokenizer, case.target)
        if target_id is None:
            continue
        encoded, mask_index = encode_prompt(tokenizer, case.prompt.replace("[MASK]", tokenizer.mask_token))
        hidden_states, final_logits = hidden_states_and_logits(model, encoded)
        final_mask_logits = final_logits[0, mask_index, :].detach().cpu().numpy().astype(np.float64)
        top_ids, top_probs = topk_distribution(final_mask_logits, top_k)
        top1_id = int(top_ids[0])
        observed_condition = "correct" if top1_id == int(target_id) else "error"
        sorted_all = np.argsort(final_mask_logits)[::-1]
        target_rank = int(np.where(sorted_all == target_id)[0][0] + 1)
        prompt_rows.append(
            {
                "prompt_id": prompt_id,
                "prompt": case.prompt,
                "target": case.target,
                "target_id": target_id,
                "topic": case.topic,
                "fact_id": case.fact_id,
                "template_id": case.template_id,
                "observed_condition": observed_condition,
                "target_rank_final": target_rank,
                "final_top1_token": tokenizer.convert_ids_to_tokens([top1_id])[0],
                "final_top1_prob_topk": float(top_probs[0]),
                "final_top8_tokens": " ".join(tokenizer.convert_ids_to_tokens([int(v) for v in top_ids[:8]])),
            }
        )
        records.append(
            {
                "prompt_id": prompt_id,
                "mask_index": mask_index,
                "hidden_states": hidden_states,
            }
        )
    return pd.DataFrame(prompt_rows), records


def layer_logits(model, hidden: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        logits = mlm_logits_from_hidden(model, hidden.view(1, 1, -1))[0, 0, :]
    return logits.detach().cpu().numpy().astype(np.float64)


def compute_scores_and_hidden_features(
    tokenizer,
    model,
    prompt_table: pd.DataFrame,
    records: list[dict],
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    num_layers = len(records[0]["hidden_states"]) - 1
    hidden_dim = records[0]["hidden_states"][0].shape[-1]
    rng = np.random.default_rng(args.random_seed)
    random_bases = {
        layer: [
            orthonormalize(rng.normal(size=(hidden_dim, 2)), 2)
            for _ in range(args.random_subspaces)
        ]
        for layer in range(num_layers + 1)
    }
    hidden_random_bases = {
        layer: orthonormalize(
            rng.normal(size=(hidden_dim, args.hidden_random_projections)),
            args.hidden_random_projections,
        )
        for layer in range(num_layers + 1)
    }
    score_rows: list[dict] = []
    hidden_rows: list[dict] = []
    for record in records:
        meta = prompt_table[prompt_table["prompt_id"].eq(record["prompt_id"])].iloc[0]
        hidden_states = record["hidden_states"]
        mask_index = record["mask_index"]
        hidden_row = {
            "prompt_id": int(record["prompt_id"]),
            "observed_condition": meta["observed_condition"],
            "topic": meta["topic"],
            "fact_id": meta["fact_id"],
            "template_id": meta["template_id"],
        }
        for layer in range(num_layers + 1):
            hidden = hidden_states[layer][0, mask_index, :].detach()
            hidden_vec = hidden.cpu().numpy().astype(np.float64)
            hidden_row[f"state_norm_l{layer}"] = float(np.linalg.norm(hidden_vec))
            for idx, value in enumerate(hidden_vec @ hidden_random_bases[layer]):
                hidden_row[f"state_random_l{layer}_{idx}"] = float(value)
            if layer > 0:
                prev_vec = hidden_states[layer - 1][0, mask_index, :].detach().cpu().numpy().astype(np.float64)
                delta_vec = hidden_vec - prev_vec
                hidden_row[f"delta_norm_l{layer}"] = float(np.linalg.norm(delta_vec))
                for idx, value in enumerate(delta_vec @ hidden_random_bases[layer]):
                    hidden_row[f"delta_random_l{layer}_{idx}"] = float(value)

            logits = layer_logits(model, hidden)
            token_ids, probs = topk_distribution(logits, args.top_k)
            jac = mlm_head_jacobian(model, hidden, token_ids)
            bases = {
                "state_1d": normalize(hidden_vec)[:, None],
                "full": np.eye(hidden_dim),
            }
            if layer > 0:
                bases["delta_1d"] = normalize(delta_vec)[:, None]
            for idx, basis in enumerate(random_bases[layer]):
                bases[f"random_2_{idx}"] = basis
            for subspace, basis in bases.items():
                score = fisher_scores_for_sample(probs, jac @ basis)
                score_rows.append(
                    {
                        "prompt_id": int(record["prompt_id"]),
                        "layer": layer,
                        "subspace": subspace,
                        "subspace_dim": basis.shape[1],
                        "observed_condition": meta["observed_condition"],
                        "topic": meta["topic"],
                        "fact_id": meta["fact_id"],
                        "template_id": meta["template_id"],
                        "entropy_topk": score.entropy,
                        "varentropy_topk": score.varentropy,
                        "accessible_varentropy": score.accessible_varentropy,
                        "rho": score.rho,
                        "grad_norm_sq": score.grad_norm_sq,
                        "trace_fim": score.trace_fim,
                        "top1_prob_topk_layer": float(probs[0]),
                    }
                )
        hidden_rows.append(hidden_row)
    return pd.DataFrame(score_rows), pd.DataFrame(hidden_rows)


def add_summary_columns(features: pd.DataFrame, prefixes: list[str], suffix: str) -> list[str]:
    created: list[str] = []
    for prefix in prefixes:
        cols = [c for c in features.columns if c.startswith(prefix)]
        if not cols:
            continue
        for stat_name, values in {
            "mean": features[cols].mean(axis=1),
            "std": features[cols].std(axis=1).fillna(0.0),
            "min": features[cols].min(axis=1),
            "max": features[cols].max(axis=1),
        }.items():
            col = f"{prefix.rstrip('_')}_{suffix}_{stat_name}"
            features[col] = values
            created.append(col)
    return created


def make_prompt_features(scores: pd.DataFrame, hidden_features: pd.DataFrame) -> pd.DataFrame:
    full = scores[scores["subspace"].eq("full")]
    scalar = full.pivot_table(
        index=["prompt_id", "observed_condition", "topic", "fact_id", "template_id"],
        columns="layer",
        values=["entropy_topk", "varentropy_topk", "trace_fim", "grad_norm_sq", "top1_prob_topk_layer"],
        aggfunc="mean",
    )
    scalar.columns = [f"{metric}_l{layer}" for metric, layer in scalar.columns]
    non_full = scores[~scores["subspace"].eq("full")]
    rho = non_full.pivot_table(
        index=["prompt_id", "observed_condition", "topic", "fact_id", "template_id"],
        columns=["layer", "subspace"],
        values="rho",
        aggfunc="mean",
    )
    rho.columns = [f"rho_l{layer}_{subspace}" for layer, subspace in rho.columns]
    features = scalar.join(rho).reset_index()
    hidden_cols = [c for c in hidden_features.columns if c == "prompt_id" or c.startswith(("state_norm_", "delta_norm_", "state_random_", "delta_random_"))]
    features = features.merge(hidden_features[hidden_cols], on="prompt_id", how="left")

    scalar_prefixes = ["entropy_topk_l", "varentropy_topk_l", "trace_fim_l", "grad_norm_sq_l", "top1_prob_topk_layer_l"]
    add_summary_columns(features, scalar_prefixes, "profile")
    add_summary_columns(features, ["rho_l"], "all")
    add_summary_columns(features, ["state_norm_l", "delta_norm_l"], "profile")
    add_summary_columns(features, ["state_random_l", "delta_random_l"], "profile")
    return features


def feature_sets(features: pd.DataFrame) -> dict[str, list[str]]:
    scalar_layer = [
        c
        for c in features.columns
        if c.startswith(("entropy_topk_l", "varentropy_topk_l", "trace_fim_l", "grad_norm_sq_l", "top1_prob_topk_layer_l"))
        and "_profile_" not in c
    ]
    layers = sorted({int(c.rsplit("_l", 1)[1]) for c in scalar_layer if c.rsplit("_l", 1)[1].isdigit()})
    final_layer = max(layers)
    scalar_final = [c for c in scalar_layer if c.endswith(f"_l{final_layer}")]
    output_final = [c for c in scalar_final if c.startswith(("entropy_topk_", "varentropy_topk_", "top1_prob_topk_layer_"))]
    scalar_summary = [c for c in features.columns if any(c.startswith(prefix) and "_profile_" in c for prefix in ["entropy_topk_l", "varentropy_topk_l", "trace_fim_l", "grad_norm_sq_l", "top1_prob_topk_layer_l"])]

    rho_state = [c for c in features.columns if c.startswith("rho_l") and "state_1d" in c]
    rho_delta = [c for c in features.columns if c.startswith("rho_l") and "delta_1d" in c]
    rho_random = [c for c in features.columns if c.startswith("rho_l") and "random_2_" in c]
    rho_summary = [c for c in features.columns if c.startswith("rho_l_all_")]
    hidden_norm = [c for c in features.columns if c.startswith(("state_norm_l", "delta_norm_l")) and "_profile_" not in c]
    hidden_random = [c for c in features.columns if c.startswith(("state_random_l", "delta_random_l")) and "_profile_" not in c]
    hidden_summary = [c for c in features.columns if c.startswith(("state_norm_l_profile_", "delta_norm_l_profile_", "state_random_l_profile_", "delta_random_l_profile_"))]
    return {
        "output_final_scalar": output_final,
        "scalar_final": scalar_final,
        "scalar_summary": scalar_summary,
        "scalar_layer_profile": scalar_layer,
        "rho_state": rho_state,
        "rho_delta": rho_delta,
        "rho_state_delta": rho_state + rho_delta,
        "rho_random": rho_random,
        "rho_summary": rho_summary,
        "scalar_final_plus_delta": scalar_final + rho_delta,
        "scalar_layer_plus_delta": scalar_layer + rho_delta,
        "scalar_layer_plus_state_delta": scalar_layer + rho_state + rho_delta,
        "scalar_layer_plus_random": scalar_layer + rho_random,
        "hidden_norm_profile": hidden_norm,
        "hidden_random_profile": hidden_random,
        "hidden_summary": hidden_summary,
        "hidden_all": hidden_norm + hidden_random,
    }


def classifier(c: float):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="liblinear",
            C=c,
        ),
    )


def evaluate_splits(
    features: pd.DataFrame,
    fsets: dict[str, list[str]],
    split_kind: str,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = features["observed_condition"].eq("error").astype(int).values
    if min(np.bincount(y)) < 4:
        return pd.DataFrame(), pd.DataFrame()
    if split_kind == "random":
        splitter = StratifiedShuffleSplit(
            n_splits=args.splits,
            test_size=args.test_size,
            random_state=args.random_seed,
        )
        splits = list(splitter.split(features, y))
    elif split_kind == "fact_group":
        splitter = GroupShuffleSplit(
            n_splits=args.splits,
            test_size=args.test_size,
            random_state=args.random_seed,
        )
        groups = features["fact_id"].astype(str).values
        splits = [
            (train_idx, test_idx)
            for train_idx, test_idx in splitter.split(features, y, groups)
            if len(np.unique(y[train_idx])) == 2 and len(np.unique(y[test_idx])) == 2
        ]
    else:
        raise ValueError(split_kind)

    rows: list[dict] = []
    predictions: list[dict] = []
    for split_id, (train_idx, test_idx) in enumerate(splits):
        for name, cols in fsets.items():
            if not cols or features[cols].isna().any().any():
                continue
            clf = classifier(args.logreg_c)
            clf.fit(features.iloc[train_idx][cols].values, y[train_idx])
            predicted = clf.predict_proba(features.iloc[test_idx][cols].values)[:, 1]
            auc = float(roc_auc_score(y[test_idx], predicted))
            rows.append(
                {
                    "split_kind": split_kind,
                    "split": split_id,
                    "feature_set": name,
                    "n": len(features),
                    "positive": int(y.sum()),
                    "test_positive": int(y[test_idx].sum()),
                    "n_features": len(cols),
                    "auc": auc,
                }
            )
            for prompt_id, score, label in zip(features.iloc[test_idx]["prompt_id"], predicted, y[test_idx]):
                predictions.append(
                    {
                        "split_kind": split_kind,
                        "split": split_id,
                        "feature_set": name,
                        "prompt_id": int(prompt_id),
                        "score": float(score),
                        "label_error": int(label),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(predictions)


def evaluate_leave_topic_out(features: pd.DataFrame, fsets: dict[str, list[str]], args: argparse.Namespace) -> pd.DataFrame:
    y = features["observed_condition"].eq("error").astype(int).values
    groups = features["topic"].astype(str).values
    rows: list[dict] = []
    if len(np.unique(groups)) < 2 or min(np.bincount(y)) < 4:
        return pd.DataFrame()
    logo = LeaveOneGroupOut()
    for feature_set, cols in fsets.items():
        if not cols or features[cols].isna().any().any():
            continue
        fold_scores: list[float] = []
        fold_topics: list[str] = []
        for train_idx, test_idx in logo.split(features, y, groups):
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                continue
            clf = classifier(args.logreg_c)
            clf.fit(features.iloc[train_idx][cols].values, y[train_idx])
            predicted = clf.predict_proba(features.iloc[test_idx][cols].values)[:, 1]
            fold_scores.append(float(roc_auc_score(y[test_idx], predicted)))
            fold_topics.append(str(groups[test_idx][0]))
        if fold_scores:
            rows.append(
                {
                    "feature_set": feature_set,
                    "n_folds": len(fold_scores),
                    "held_out_topics": ",".join(fold_topics),
                    "auc_mean": float(np.mean(fold_scores)),
                    "auc_std": float(np.std(fold_scores)),
                    "auc_min": float(np.min(fold_scores)),
                    "auc_max": float(np.max(fold_scores)),
                }
            )
    return pd.DataFrame(rows)


def summarize_split_scores(split_scores: pd.DataFrame) -> pd.DataFrame:
    if split_scores.empty:
        return split_scores
    return (
        split_scores.groupby(["split_kind", "feature_set"], as_index=False)
        .agg(
            n=("n", "first"),
            positive=("positive", "first"),
            n_features=("n_features", "first"),
            auc_mean=("auc", "mean"),
            auc_std=("auc", "std"),
            auc_min=("auc", "min"),
            auc_max=("auc", "max"),
        )
        .sort_values(["split_kind", "auc_mean"], ascending=[True, False])
    )


def paired_deltas(split_scores: pd.DataFrame) -> pd.DataFrame:
    if split_scores.empty:
        return split_scores
    pairs = [
        ("rho_delta", "scalar_final"),
        ("rho_delta", "scalar_layer_profile"),
        ("rho_state_delta", "scalar_final"),
        ("rho_state_delta", "scalar_layer_profile"),
        ("rho_state_delta", "hidden_all"),
        ("scalar_layer_plus_delta", "scalar_layer_profile"),
        ("scalar_layer_plus_state_delta", "scalar_layer_profile"),
        ("rho_random", "rho_state_delta"),
        ("hidden_all", "rho_state_delta"),
    ]
    rows: list[dict] = []
    wide = split_scores.pivot_table(index=["split_kind", "split"], columns="feature_set", values="auc")
    for split_kind, local in wide.groupby(level="split_kind"):
        frame = local.reset_index()
        for left, right in pairs:
            if left not in frame.columns or right not in frame.columns:
                continue
            delta = frame[left] - frame[right]
            rows.append(
                {
                    "split_kind": split_kind,
                    "left": left,
                    "right": right,
                    "delta_mean": float(delta.mean()),
                    "delta_std": float(delta.std()),
                    "delta_p05": float(delta.quantile(0.05)),
                    "delta_p50": float(delta.quantile(0.50)),
                    "delta_p95": float(delta.quantile(0.95)),
                    "win_rate": float((delta > 0).mean()),
                }
            )
    return pd.DataFrame(rows)


def write_report(
    out_dir: Path,
    model_name: str,
    prompt_table: pd.DataFrame,
    scores: pd.DataFrame,
    summary: pd.DataFrame,
    deltas: pd.DataFrame,
    leave_topic: pd.DataFrame,
) -> None:
    non_full = scores[scores["subspace"] != "full"]
    fixed = non_full.groupby(["prompt_id", "layer"])
    distinct = pd.DataFrame(
        {
            "rho_range": fixed["rho"].max() - fixed["rho"].min(),
            "entropy_range": fixed["entropy_topk"].max() - fixed["entropy_topk"].min(),
        }
    ).describe()
    full = scores[scores["subspace"].eq("full")].groupby("layer")["rho"].describe()
    key_sets = [
        "output_final_scalar",
        "scalar_final",
        "scalar_layer_profile",
        "rho_delta",
        "rho_state_delta",
        "rho_random",
        "scalar_layer_plus_delta",
        "scalar_layer_plus_state_delta",
        "hidden_all",
    ]
    key_summary = summary[summary["feature_set"].isin(key_sets)] if not summary.empty else summary
    key_deltas = deltas[
        deltas.apply(
            lambda r: (r["left"], r["right"])
            in {
                ("rho_delta", "scalar_layer_profile"),
                ("rho_state_delta", "scalar_layer_profile"),
                ("scalar_layer_plus_delta", "scalar_layer_profile"),
                ("rho_random", "rho_state_delta"),
                ("hidden_all", "rho_state_delta"),
            },
            axis=1,
        )
    ] if not deltas.empty else deltas
    lines = [
        "# Factual Error Deep Benchmark",
        "",
        f"Model: `{model_name}`",
        "",
        "## Prompt Counts",
        "```text",
        prompt_table.groupby(["topic", "observed_condition"]).size().to_string(),
        "```",
        "",
        "## Key Held-Out AUCs",
        "```text",
        key_summary.to_string(index=False),
        "```",
        "",
        "## Key Paired Deltas",
        "```text",
        key_deltas.to_string(index=False),
        "```",
        "",
        "## Leave-Topic-Out",
        "```text",
        leave_topic.to_string(index=False),
        "```",
        "",
        "## Within Prompt/Layer Distinctness",
        "```text",
        distinct.to_string(),
        "```",
        "",
        "## Full-Space Sanity",
        "```text",
        full.to_string(),
        "```",
        "",
        "## Example Errors",
        "```text",
        prompt_table[prompt_table["observed_condition"].eq("error")][
            ["topic", "prompt", "target", "final_top1_token", "target_rank_final", "final_top8_tokens"]
        ].head(40).to_string(index=False),
        "```",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    tokenizer = BertTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    model = AutoModelForMaskedLM.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        attn_implementation="eager",
    )
    model.eval()

    cases = factual_error_cases()
    if args.max_prompts > 0:
        cases = cases[: args.max_prompts]

    prompt_table, records = collect_records(tokenizer, model, cases, args.top_k)
    if prompt_table.empty or prompt_table["observed_condition"].nunique() < 2:
        raise RuntimeError("Need both correct and error prompts after filtering.")
    scores, hidden_features = compute_scores_and_hidden_features(tokenizer, model, prompt_table, records, args)
    features = make_prompt_features(scores, hidden_features)
    fsets = feature_sets(features)

    random_scores, random_predictions = evaluate_splits(features, fsets, "random", args)
    group_scores, group_predictions = evaluate_splits(features, fsets, "fact_group", args)
    split_scores = pd.concat([random_scores, group_scores], ignore_index=True)
    predictions = pd.concat([random_predictions, group_predictions], ignore_index=True)
    summary = summarize_split_scores(split_scores)
    deltas = paired_deltas(split_scores)
    leave_topic = evaluate_leave_topic_out(features, fsets, args)

    prompt_table.to_csv(args.out_dir / "prompts.csv", index=False)
    scores.to_csv(args.out_dir / "sample_scores.csv", index=False)
    hidden_features.to_csv(args.out_dir / "hidden_features.csv", index=False)
    features.to_csv(args.out_dir / "prompt_features.csv", index=False)
    split_scores.to_csv(args.out_dir / "split_scores.csv", index=False)
    predictions.to_csv(args.out_dir / "split_predictions.csv", index=False)
    summary.to_csv(args.out_dir / "feature_set_summary.csv", index=False)
    deltas.to_csv(args.out_dir / "paired_deltas.csv", index=False)
    leave_topic.to_csv(args.out_dir / "leave_topic_out.csv", index=False)
    write_report(args.out_dir, args.model, prompt_table, scores, summary, deltas, leave_topic)
    report_text = (args.out_dir / "report.md").read_text(encoding="utf-8")
    print(report_text.encode("ascii", errors="replace").decode("ascii"))


if __name__ == "__main__":
    main()
