from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in [SRC, SCRIPTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from accessible_varentropy.metrics import softmax_np
from run_decoder_uncertainty_steering import (
    entropy_varentropy,
    fisher_geometry,
    get_final_norm,
    orthogonal_basis_projection,
)


OUT = ROOT / "applications" / "09_confidence_repair_abstention"
ANSWER_LABELS = ["A", "B", "C", "D"]
CONFIDENCE_LABELS = {"low": 0.25, "medium": 0.5, "high": 0.85}


@dataclass(frozen=True)
class QAItem:
    item_id: int
    task_type: str
    question: str
    choices: tuple[str, str, str, str]
    answer: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--layers", default="auto")
    parser.add_argument("--pca-dim", type=int, default=1)
    parser.add_argument("--eps", default="0.05,0.1,0.2,0.4")
    parser.add_argument("--high-confidence-threshold", type=float, default=0.70)
    parser.add_argument("--low-confidence-threshold", type=float, default=0.45)
    parser.add_argument("--max-items", type=int, default=240)
    parser.add_argument("--bootstrap", type=int, default=500)
    parser.add_argument("--self-consistency-prompts", type=int, default=3)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--torch-dtype", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260603)
    return parser.parse_args()


def ensure_dirs(out_dir: Path) -> None:
    for rel in ["config", "outputs", "reports"]:
        (out_dir / rel).mkdir(parents=True, exist_ok=True)


def resolve_torch_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def move_batch(encoded: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in encoded.items()}


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_").lower()


def token_id_for_label(tokenizer, label: str) -> int:
    variants = [f" {label}", label, f"\n{label}", f"{label}."]
    for variant in variants:
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        if len(encoded) == 1:
            return int(encoded[0])
    for variant in variants:
        encoded = tokenizer.encode(variant, add_special_tokens=False)
        if encoded:
            return int(encoded[-1])
    raise RuntimeError(f"Could not map label {label!r} to a token.")


def load_decoder(args: argparse.Namespace):
    device = resolve_device(args.device)
    dtype = resolve_torch_dtype(args.torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    kwargs = {
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
        "torch_dtype": dtype,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, attn_implementation="eager", **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    model.to(device)
    model.eval()
    print(f"confidence_repair_device {args.model} {device}", flush=True)
    return tokenizer, model, get_final_norm(model), model.get_output_embeddings()


def build_qa_items() -> list[QAItem]:
    raw = [
        ("math", "If a train travels 60 miles in 1.5 hours, what is its average speed?", ("30 mph", "40 mph", "45 mph", "90 mph"), "B"),
        ("math", "A box has 3 red balls and 2 blue balls. What fraction are blue?", ("2/5", "3/5", "1/2", "2/3"), "A"),
        ("math", "What is 17 + 28?", ("35", "45", "46", "55"), "B"),
        ("math", "If 4 notebooks cost 12 dollars, how much do 7 notebooks cost at the same rate?", ("18", "19", "21", "24"), "C"),
        ("math", "A rectangle is 8 by 5. What is its area?", ("13", "26", "40", "80"), "C"),
        ("math", "What is 15 percent of 200?", ("15", "20", "30", "45"), "C"),
        ("math", "If x + 7 = 12, what is x?", ("3", "5", "7", "19"), "B"),
        ("math", "Which number is prime?", ("21", "27", "29", "33"), "C"),
        ("science", "What gas do plants primarily take in for photosynthesis?", ("Oxygen", "Carbon dioxide", "Nitrogen", "Hydrogen"), "B"),
        ("science", "What organ pumps blood through the human body?", ("Lung", "Liver", "Heart", "Kidney"), "C"),
        ("science", "Water freezes at what temperature at sea level?", ("0 C", "10 C", "32 C", "100 C"), "A"),
        ("science", "Which planet is known as the Red Planet?", ("Venus", "Mars", "Jupiter", "Mercury"), "B"),
        ("science", "What force pulls objects toward Earth?", ("Friction", "Magnetism", "Gravity", "Evaporation"), "C"),
        ("science", "What particle has a negative electric charge?", ("Proton", "Neutron", "Electron", "Photon"), "C"),
        ("commonsense", "Which object is best for cutting paper?", ("Spoon", "Scissors", "Pillow", "Cup"), "B"),
        ("commonsense", "If it is raining, what should you bring to stay dry?", ("Umbrella", "Sunglasses", "Sandals", "Notebook"), "A"),
        ("commonsense", "Where would you usually store frozen food?", ("Oven", "Freezer", "Desk", "Bookshelf"), "B"),
        ("commonsense", "Which item is typically used to write on a whiteboard?", ("Marker", "Fork", "Blanket", "Hammer"), "A"),
        ("commonsense", "What do people usually use to unlock a door?", ("Key", "Plate", "Soap", "Pencil"), "A"),
        ("logic", "All squares are rectangles. This shape is a square. What follows?", ("It is a rectangle", "It is a circle", "It is not a rectangle", "No conclusion"), "A"),
        ("logic", "If all mammals are animals and whales are mammals, whales are what?", ("Plants", "Animals", "Minerals", "Machines"), "B"),
        ("logic", "A is taller than B. B is taller than C. Who is tallest?", ("A", "B", "C", "Cannot tell"), "A"),
        ("logic", "If the switch is off, the lamp is dark. The lamp is not dark. What can you infer?", ("Switch is off", "Switch is on", "Lamp is broken", "No inference"), "B"),
        ("logic", "Only employees can enter. Maria entered. What is the best inference?", ("Maria is an employee", "Maria is a visitor", "Maria is outside", "Maria is absent"), "A"),
        ("trick", "A bat and ball cost 1.10 dollars. The bat costs 1 dollar more than the ball. What does the ball cost?", ("5 cents", "10 cents", "15 cents", "1 dollar"), "A"),
        ("trick", "If you overtake the person in second place, what place are you in?", ("First", "Second", "Third", "Last"), "B"),
        ("trick", "A farmer has 15 sheep and all but 8 run away. How many are left?", ("7", "8", "15", "23"), "B"),
        ("trick", "How many months have 28 days?", ("1", "2", "11", "12"), "D"),
        ("trick", "Before Mount Everest was discovered, what was the tallest mountain?", ("K2", "Everest", "Kangchenjunga", "Unknown"), "B"),
        ("trick", "A doctor gives you 3 pills and says take one every 30 minutes. How long until all are taken?", ("30 minutes", "60 minutes", "90 minutes", "120 minutes"), "B"),
        ("knowledge", "Who wrote the play Hamlet?", ("Charles Dickens", "William Shakespeare", "Jane Austen", "Mark Twain"), "B"),
        ("knowledge", "What is the capital of France?", ("Berlin", "Madrid", "Paris", "Rome"), "C"),
        ("knowledge", "Which language is primarily spoken in Brazil?", ("Spanish", "Portuguese", "French", "Italian"), "B"),
        ("knowledge", "What is H2O commonly called?", ("Salt", "Water", "Oxygen", "Hydrogen"), "B"),
        ("knowledge", "Which continent is Egypt in?", ("Asia", "Europe", "Africa", "Australia"), "C"),
        ("knowledge", "What instrument has black and white keys?", ("Guitar", "Piano", "Drum", "Flute"), "B"),
    ]
    base = [QAItem(i, task, question, choices, answer) for i, (task, question, choices, answer) in enumerate(raw)]
    expanded: list[QAItem] = []
    item_id = 0
    for item in base:
        wrong_labels = [label for label in ANSWER_LABELS if label != item.answer]
        for variant in range(6):
            choices = item.choices
            question = item.question
            answer = item.answer
            if variant == 1:
                question = "Solve carefully; a common first guess is often wrong. " + question
            elif variant == 2:
                decoy = wrong_labels[(item.item_id + variant) % len(wrong_labels)]
                question = f"A previous assistant guessed {decoy}, but you should verify independently. {question}"
            elif variant == 3:
                question = "Do not explain. Choose the best answer. " + question
            elif variant == 4:
                order = [2, 0, 3, 1]
                old_to_new = {ANSWER_LABELS[old]: ANSWER_LABELS[new] for new, old in enumerate(order)}
                choices = tuple(item.choices[i] for i in order)
                answer = old_to_new[item.answer]
            elif variant == 5:
                order = [1, 3, 0, 2]
                old_to_new = {ANSWER_LABELS[old]: ANSWER_LABELS[new] for new, old in enumerate(order)}
                choices = tuple(item.choices[i] for i in order)
                answer = old_to_new[item.answer]
                question = "The options may be in an unusual order. " + question
            expanded.append(QAItem(item_id, item.task_type, question, choices, answer))
            item_id += 1
    return expanded


def qa_prompt(item: QAItem, variant: int = 0) -> str:
    styles = [
        "Answer with only the letter A, B, C, or D.",
        "Choose the single best option. Reply with just A, B, C, or D.",
        "Select the correct answer letter only.",
    ]
    choices = "\n".join(f"{label}. {text}" for label, text in zip(ANSWER_LABELS, item.choices))
    return f"{styles[variant % len(styles)]}\nQuestion: {item.question}\n{choices}\nAnswer:"


def confidence_prompt(item: QAItem, answer: str) -> str:
    choices = "\n".join(f"{label}. {text}" for label, text in zip(ANSWER_LABELS, item.choices))
    return (
        "A model answered a multiple-choice question. Rate confidence as low, medium, or high. "
        "Reply with only one word.\n"
        f"Question: {item.question}\n{choices}\nModel answer: {answer}\nConfidence:"
    )


def parse_layers(spec: str, n_layers: int) -> list[int]:
    if spec == "auto":
        return sorted({max(1, n_layers // 2), max(1, n_layers - 1)})
    return sorted({int(part) for part in spec.split(",") if part.strip()})


def pca_basis(matrix: np.ndarray, k: int) -> np.ndarray:
    max_k = min(int(k), matrix.shape[1], max(1, matrix.shape[0] - 1))
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[:max_k].T.astype(np.float64)


def orthonormalize(matrix: np.ndarray, k: int) -> np.ndarray:
    q, _r = np.linalg.qr(matrix)
    return q[:, : min(k, q.shape[1])].astype(np.float64)


def selected_logits(hidden: torch.Tensor, norm, lm_head, token_ids: torch.Tensor) -> torch.Tensor:
    x = norm(hidden.reshape(1, 1, -1))[0, 0, :]
    weight = lm_head.weight[token_ids]
    logits = weight @ x
    bias = getattr(lm_head, "bias", None)
    if bias is not None:
        logits = logits + bias[token_ids]
    return logits


def jacobian_selected(hidden: torch.Tensor, norm, lm_head, token_ids: torch.Tensor) -> np.ndarray:
    base = hidden.detach().clone().requires_grad_(True)

    def fn(vec: torch.Tensor) -> torch.Tensor:
        return selected_logits(vec, norm, lm_head, token_ids).float()

    jac = torch.autograd.functional.jacobian(fn, base, create_graph=False, strict=False, vectorize=True)
    return jac.detach().float().cpu().numpy().astype(np.float64)


def evaluate_hidden(hidden: torch.Tensor, direction: np.ndarray, eps: float, norm, lm_head, token_ids: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    step = torch.as_tensor(float(eps) * direction, dtype=hidden.dtype, device=hidden.device)
    with torch.no_grad():
        logits = selected_logits(hidden + step, norm, lm_head, token_ids).detach().float().cpu().numpy().astype(np.float64)
    probs = softmax_np(logits[None, :])[0].astype(np.float64)
    return logits, probs


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0)
    q = np.clip(np.asarray(q, dtype=np.float64), eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    return float(
        0.5 * np.sum(p * (np.log(p) - np.log(m)))
        + 0.5 * np.sum(q * (np.log(q) - np.log(m)))
    )


def brier_multiclass(probs: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    y = np.zeros((len(labels), n_classes), dtype=np.float64)
    y[np.arange(len(labels)), labels] = 1.0
    return float(np.mean(np.sum((probs - y) ** 2, axis=1)))


def expected_calibration_error(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(bins[:-1], bins[1:]):
        mask = (conf >= low) & (conf < high if high < 1.0 else conf <= high)
        if np.any(mask):
            ece += float(mask.mean()) * abs(float(conf[mask].mean()) - float(correct[mask].mean()))
    return float(ece)


def calibration_metrics(
    df: pd.DataFrame,
    prob_cols: list[str],
    label_col: str = "gold_idx",
    high_confidence_threshold: float = 0.70,
) -> dict[str, float]:
    probs = df[prob_cols].to_numpy(dtype=float)
    labels = df[label_col].to_numpy(dtype=int)
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = (pred == labels).astype(int)
    return {
        "accuracy": float(correct.mean()),
        "mean_confidence": float(conf.mean()),
        "overconfident_wrong_rate": float(((correct == 0) & (conf >= high_confidence_threshold)).mean()),
        "nll": float(log_loss(labels, probs, labels=list(range(len(prob_cols))))),
        "brier": brier_multiclass(probs, labels, len(prob_cols)),
        "ece": expected_calibration_error(conf, correct),
    }


def temperature_scaled(probs: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probs, 1e-12, 1.0))
    return softmax_np((logits / float(temperature))[None, :])[0]


def build_records(args: argparse.Namespace, tokenizer, model, norm, lm_head, items: list[QAItem]) -> tuple[pd.DataFrame, list[dict[str, object]], list[int]]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    answer_ids = [token_id_for_label(tokenizer, label) for label in ANSWER_LABELS]
    conf_ids = [token_id_for_label(tokenizer, label) for label in CONFIDENCE_LABELS]
    token_ids_t = torch.as_tensor(answer_ids, dtype=torch.long, device=device)
    conf_ids_t = torch.as_tensor(conf_ids, dtype=torch.long, device=device)
    with torch.no_grad():
        probe = model(
            **move_batch(tokenizer("probe", return_tensors="pt", add_special_tokens=False), device),
            output_hidden_states=True,
            use_cache=False,
        )
    layers = parse_layers(args.layers, len(probe.hidden_states) - 1)
    rows: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for idx, item in enumerate(items[: args.max_items]):
        print(f"qa_item {idx + 1}/{len(items)} {item.task_type}:{item.item_id}", flush=True)
        encoded = move_batch(tokenizer(qa_prompt(item), return_tensors="pt", add_special_tokens=False), device)
        with torch.no_grad():
            out = model(**encoded, output_hidden_states=True, use_cache=False)
        logits = out.logits[0, -1, answer_ids].detach().float().cpu().numpy().astype(np.float64)
        probs = softmax_np(logits[None, :])[0]
        pred_idx = int(np.argmax(probs))
        gold_idx = ANSWER_LABELS.index(item.answer)
        sc = []
        for variant in range(args.self_consistency_prompts):
            enc_v = move_batch(tokenizer(qa_prompt(item, variant), return_tensors="pt", add_special_tokens=False), device)
            with torch.no_grad():
                out_v = model(**enc_v, use_cache=False)
            probs_v = softmax_np(out_v.logits[0, -1, answer_ids].detach().float().cpu().numpy()[None, :])[0]
            sc.append(int(np.argmax(probs_v)))
        conf_enc = move_batch(tokenizer(confidence_prompt(item, ANSWER_LABELS[pred_idx]), return_tensors="pt", add_special_tokens=False), device)
        with torch.no_grad():
            conf_out = model(**conf_enc, use_cache=False)
        conf_probs = softmax_np(conf_out.logits[0, -1, conf_ids].detach().float().cpu().numpy()[None, :])[0]
        verbalized = float(sum(prob * val for prob, val in zip(conf_probs, CONFIDENCE_LABELS.values())))
        entropy, varentropy = entropy_varentropy(probs)
        row_common = {
            "model": args.model,
            "item_id": item.item_id,
            "task_type": item.task_type,
            "question": item.question,
            "answer": item.answer,
            "gold_idx": gold_idx,
            "pred_answer": ANSWER_LABELS[pred_idx],
            "pred_idx": pred_idx,
            "correct": int(pred_idx == gold_idx),
            "confidence": float(probs[pred_idx]),
            "gold_probability": float(probs[gold_idx]),
            "entropy": entropy,
            "varentropy": varentropy,
            "margin": float(np.sort(probs)[-1] - np.sort(probs)[-2]),
            "self_consistency": float(max(sc.count(i) for i in range(len(ANSWER_LABELS))) / max(1, len(sc))),
            "verbalized_confidence": verbalized,
            **{f"p_{label}": float(prob) for label, prob in zip(ANSWER_LABELS, probs)},
        }
        rows.append(row_common)
        hidden_states = out.hidden_states
        for layer in layers:
            hidden = hidden_states[layer][0, -1, :].detach().cpu().float()
            next_hidden = hidden_states[min(layer + 1, len(hidden_states) - 1)][0, -1, :].detach().cpu().float()
            records.append({**row_common, "layer": int(layer), "hidden": hidden, "next_hidden": next_hidden, "answer_ids": answer_ids, "dtype": str(dtype)})
    base = pd.DataFrame(rows)
    base["confidence_bin"] = np.select(
        [
            base["confidence"] >= args.high_confidence_threshold,
            base["confidence"] <= args.low_confidence_threshold,
        ],
        ["high", "low"],
        default="mid",
    )
    base["outcome_group"] = np.where(base["correct"] == 1, "correct_", "wrong_") + base["confidence_bin"]
    return base, records, answer_ids


def route_bases(records: list[dict[str, object]], pca_dim: int, seed: int) -> dict[tuple[str, int], list[tuple[str, np.ndarray]]]:
    rng = np.random.default_rng(seed)
    out: dict[tuple[str, int], list[tuple[str, np.ndarray]]] = {}
    frame = pd.DataFrame([{"idx": i, "model": r["model"], "layer": r["layer"]} for i, r in enumerate(records)])
    for (model_name, layer), group in frame.groupby(["model", "layer"]):
        layer_records = [records[int(i)] for i in group["idx"]]
        hidden = np.stack([r["hidden"].numpy().astype(np.float64) for r in layer_records])
        delta = np.stack([(r["next_hidden"] - r["hidden"]).numpy().astype(np.float64) for r in layer_records])
        dim = hidden.shape[1]
        k = min(pca_dim, dim, max(1, len(layer_records) - 1))
        out[(str(model_name), int(layer))] = [
            ("state_pca", pca_basis(hidden, k)),
            ("delta_pca", pca_basis(delta, k)),
            ("random", orthonormalize(rng.normal(size=(dim, k)), k)),
        ]
    return out


def score_and_intervene(args: argparse.Namespace, model, norm, lm_head, records: list[dict[str, object]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    bases = route_bases(records, args.pca_dim, args.seed)
    eps_values = [float(part) for part in args.eps.split(",") if part.strip()]
    score_rows = []
    repair_rows = []
    for rec in records:
        hidden = rec["hidden"].to(device=device, dtype=dtype)
        token_ids = torch.as_tensor(rec["answer_ids"], dtype=torch.long, device=device)
        logits = selected_logits(hidden, norm, lm_head, token_ids).detach().float().cpu().numpy().astype(np.float64)
        probs = softmax_np(logits[None, :])[0]
        pred = int(np.argmax(probs))
        gold = int(rec["gold_idx"])
        target = np.zeros(len(ANSWER_LABELS), dtype=np.float64)
        if pred == gold:
            target[gold] = 1.0
            objective = "increase_correct_confidence"
        else:
            target[pred] = -1.0
            target += probs
            objective = "decrease_wrong_confidence"
        target = target - target.mean()
        jac = jacobian_selected(hidden, norm, lm_head, token_ids)
        full_direction = jac.T @ target
        full_norm = float(np.linalg.norm(full_direction))
        if full_norm > 1e-12:
            full_direction = full_direction / full_norm
        geom = fisher_geometry(probs)
        routes = bases[(rec["model"], int(rec["layer"]))]
        route_scores = []
        for route_name, basis in routes:
            jb = jac @ basis
            whitened = geom.sqrt_fisher @ jb
            w_acc, rank = orthogonal_basis_projection(whitened, geom.w)
            denom = max(float(geom.w @ geom.w), 1e-12)
            rho = float((w_acc @ w_acc) / denom)
            coeff = basis.T @ (jac.T @ target)
            direction = basis @ coeff
            direction_norm = float(np.linalg.norm(direction))
            if direction_norm > 1e-12:
                direction = direction / direction_norm
            route_scores.append((route_name, rho, rank, direction, basis))
            score_rows.append(
                {
                    "model": rec["model"],
                    "item_id": rec["item_id"],
                    "task_type": rec["task_type"],
                    "layer": rec["layer"],
                    "route": route_name,
                    "rho": rho,
                    "rank": rank,
                    "objective": objective,
                    "correct": rec["correct"],
                    "confidence": rec["confidence"],
                    "outcome_group": ("correct_" if rec["correct"] else "wrong_") + "unassigned",
                }
            )
        route_scores_sorted = sorted(route_scores, key=lambda x: x[1], reverse=True)
        selected_routes = [("accessible_rho", route_scores_sorted[0][3]), ("low_rho", route_scores_sorted[-1][3]), ("projected_gradient", full_direction)]
        rng = np.random.default_rng(args.seed + int(rec["item_id"]) * 31 + int(rec["layer"]))
        random_direction = rng.normal(size=full_direction.shape)
        random_direction = random_direction / max(float(np.linalg.norm(random_direction)), 1e-12)
        selected_routes.append(("random_activation", random_direction))
        selected_routes.append(("standard_delta_pca", route_scores[1][4][:, 0] if route_scores[1][4].size else random_direction))
        for method, direction in selected_routes:
            direction = np.asarray(direction, dtype=np.float64)
            direction = direction / max(float(np.linalg.norm(direction)), 1e-12)
            for eps in eps_values:
                _after_logits, after_probs = evaluate_hidden(hidden, direction, eps, norm, lm_head, token_ids)
                after_pred = int(np.argmax(after_probs))
                before_conf = float(probs[pred])
                after_conf_same_answer = float(after_probs[pred])
                after_gold_prob = float(after_probs[gold])
                repair_rows.append(
                    {
                        "model": rec["model"],
                        "item_id": rec["item_id"],
                        "task_type": rec["task_type"],
                        "layer": rec["layer"],
                        "method": method,
                        "eps": eps,
                        "objective": objective,
                        "correct_before": int(pred == gold),
                        "pred_before": ANSWER_LABELS[pred],
                        "pred_after": ANSWER_LABELS[after_pred],
                        "gold": ANSWER_LABELS[gold],
                        "answer_preserved": int(after_pred == pred),
                        "correct_after": int(after_pred == gold),
                        "confidence_before": before_conf,
                        "confidence_after": float(after_probs[after_pred]),
                        "same_answer_confidence_after": after_conf_same_answer,
                        "gold_probability_before": float(probs[gold]),
                        "gold_probability_after": after_gold_prob,
                        "wrong_confidence_reduced": int((pred != gold) and (after_conf_same_answer < before_conf)),
                        "correct_confidence_increased": int((pred == gold) and (after_gold_prob > float(probs[gold]))),
                        "overcorrection": int((pred != gold) and (after_pred == gold) and (after_probs[after_pred] > 0.8)),
                        "semantic_drift_proxy": js_divergence(probs, after_probs),
                        **{f"p_after_{label}": float(prob) for label, prob in zip(ANSWER_LABELS, after_probs)},
                    }
                )
        for temperature in [1.25, 1.5, 2.0]:
            after_probs = temperature_scaled(probs, temperature)
            repair_rows.append(
                {
                    "model": rec["model"],
                    "item_id": rec["item_id"],
                    "task_type": rec["task_type"],
                    "layer": rec["layer"],
                    "method": f"temperature_{temperature:g}",
                    "eps": 0.0,
                    "objective": "temperature_rescale",
                    "correct_before": int(pred == gold),
                    "pred_before": ANSWER_LABELS[pred],
                    "pred_after": ANSWER_LABELS[int(np.argmax(after_probs))],
                    "gold": ANSWER_LABELS[gold],
                    "answer_preserved": int(np.argmax(after_probs) == pred),
                    "correct_after": int(np.argmax(after_probs) == gold),
                    "confidence_before": float(probs[pred]),
                    "confidence_after": float(after_probs.max()),
                    "same_answer_confidence_after": float(after_probs[pred]),
                    "gold_probability_before": float(probs[gold]),
                    "gold_probability_after": float(after_probs[gold]),
                    "wrong_confidence_reduced": int((pred != gold) and (after_probs[pred] < probs[pred])),
                    "correct_confidence_increased": int((pred == gold) and (after_probs[gold] > probs[gold])),
                    "overcorrection": 0,
                    "semantic_drift_proxy": js_divergence(probs, after_probs),
                    **{f"p_after_{label}": float(prob) for label, prob in zip(ANSWER_LABELS, after_probs)},
                }
            )
    return pd.DataFrame(score_rows), pd.DataFrame(repair_rows)


def summarize_repair(args: argparse.Namespace, base: pd.DataFrame, repair: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    threshold = float(base["confidence"].median())
    group_map = base.set_index("item_id")["outcome_group"].to_dict()
    repair = repair.copy()
    repair["outcome_group"] = repair["item_id"].map(group_map)
    summary = (
        repair.groupby(["outcome_group", "method", "eps"], dropna=False)
        .agg(
            n=("item_id", "size"),
            answer_preserved_rate=("answer_preserved", "mean"),
            correct_after_rate=("correct_after", "mean"),
            confidence_delta=("confidence_after", lambda s: float(s.mean()) - float(repair.loc[s.index, "confidence_before"].mean())),
            wrong_confidence_reduced_rate=("wrong_confidence_reduced", "mean"),
            correct_confidence_increased_rate=("correct_confidence_increased", "mean"),
            overcorrection_rate=("overcorrection", "mean"),
            semantic_drift_proxy_mean=("semantic_drift_proxy", "mean"),
            gold_probability_delta=("gold_probability_after", lambda s: float(s.mean()) - float(repair.loc[s.index, "gold_probability_before"].mean())),
        )
        .reset_index()
    )
    chosen = repair[(repair["method"] == "accessible_rho") & (repair["eps"] == repair["eps"].median())].copy()
    chosen = chosen.sort_values(["item_id", "layer"]).drop_duplicates(["item_id"], keep="last")
    after_cols = [f"p_after_{label}" for label in ANSWER_LABELS]
    temp = chosen.rename(columns={col: f"p_{label}" for col, label in zip(after_cols, ANSWER_LABELS)})
    temp["gold_idx"] = temp["gold"].map({label: i for i, label in enumerate(ANSWER_LABELS)})
    before_metrics = pd.DataFrame([{"condition": "before", **calibration_metrics(base, [f"p_{label}" for label in ANSWER_LABELS], high_confidence_threshold=args.high_confidence_threshold)}])
    after_metrics = pd.DataFrame([{"condition": "accessible_rho_after_eps_median", **calibration_metrics(temp, [f"p_{label}" for label in ANSWER_LABELS], high_confidence_threshold=args.high_confidence_threshold)}])
    metrics = pd.concat([before_metrics, after_metrics], ignore_index=True)
    return repair, summary, metrics


def selective_policy_tables(base: pd.DataFrame, scores: pd.DataFrame, repair: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    high_scores = scores.sort_values("rho", ascending=False).drop_duplicates(["item_id", "layer"])
    rho_by_item = high_scores.groupby("item_id")["rho"].mean()
    work = base.copy()
    work["rho"] = work["item_id"].map(rho_by_item)
    rep = repair[(repair["method"] == "accessible_rho") & (repair["eps"] == repair["eps"].median())]
    stable = rep.groupby("item_id")["answer_preserved"].mean()
    moved_good = rep.groupby("item_id").apply(
        lambda g: float(
            np.mean(
                np.where(
                    g["correct_before"].to_numpy(dtype=int) == 1,
                    g["correct_confidence_increased"].to_numpy(dtype=int),
                    g["wrong_confidence_reduced"].to_numpy(dtype=int),
                )
            )
        ),
        include_groups=False,
    )
    work["repair_stability"] = work["item_id"].map(stable).fillna(0.0)
    work["repair_direction_success"] = work["item_id"].map(moved_good).fillna(0.0)
    policies = {
        "confidence": work["confidence"],
        "negative_entropy": -work["entropy"],
        "margin": work["margin"],
        "self_consistency": work["self_consistency"],
        "verbalized_confidence": work["verbalized_confidence"],
        "rho": work["rho"],
        "accessible_policy": 0.35 * work["confidence"] + 0.25 * work["rho"] + 0.25 * work["repair_stability"] + 0.15 * work["repair_direction_success"],
    }
    rows = []
    curve_rows = []
    correct = work["correct"].to_numpy(dtype=int)
    for name, score in policies.items():
        values = score.to_numpy(dtype=float)
        if len(np.unique(correct)) == 2 and len(np.unique(values)) > 1:
            auroc = float(roc_auc_score(correct, values))
            auprc = float(average_precision_score(correct, values))
        else:
            auroc = float("nan")
            auprc = float("nan")
        rows.append({"policy": name, "metric": "auroc_correct", "value": auroc, "n": len(work)})
        rows.append({"policy": name, "metric": "auprc_correct", "value": auprc, "n": len(work)})
        for coverage in [0.25, 0.5, 0.75, 1.0]:
            cutoff = np.quantile(values, 1.0 - coverage)
            answer = values >= cutoff
            actual_coverage = float(answer.mean())
            risk = float(1.0 - correct[answer].mean()) if np.any(answer) else float("nan")
            curve_rows.append({"policy": name, "target_coverage": coverage, "coverage": actual_coverage, "selective_risk": risk, "answered": int(answer.sum())})
    return pd.DataFrame(rows), pd.DataFrame(curve_rows)


def bootstrap_confidence_intervals(
    base: pd.DataFrame,
    repair: pd.DataFrame,
    risk_curves: pd.DataFrame,
    *,
    n_bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    if n_bootstrap <= 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    item_ids = base["item_id"].to_numpy()
    rows = []
    for _ in range(n_bootstrap):
        sample = rng.choice(item_ids, size=len(item_ids), replace=True)
        base_s = pd.concat([base[base["item_id"] == item_id] for item_id in sample], ignore_index=True)
        repair_s = pd.concat([repair[repair["item_id"] == item_id] for item_id in sample], ignore_index=True)
        correct = base_s["correct"].to_numpy(dtype=int)
        conf = base_s["confidence"].to_numpy(dtype=float)
        rows.append(
            {
                "metric": "accuracy",
                "group": "base",
                "method": "none",
                "coverage": np.nan,
                "value": float(correct.mean()),
            }
        )
        rows.append(
            {
                "metric": "mean_confidence",
                "group": "base",
                "method": "none",
                "coverage": np.nan,
                "value": float(conf.mean()),
            }
        )
        for (group, method, eps), local in repair_s.groupby(["outcome_group", "method", "eps"], dropna=False):
            if group not in {"wrong_high", "correct_low", "correct_high", "wrong_low"}:
                continue
            rows.append(
                {
                    "metric": "answer_preserved_rate",
                    "group": group,
                    "method": method,
                    "coverage": np.nan,
                    "eps": eps,
                    "value": float(local["answer_preserved"].mean()),
                }
            )
            rows.append(
                {
                    "metric": "confidence_delta",
                    "group": group,
                    "method": method,
                    "coverage": np.nan,
                    "eps": eps,
                    "value": float(local["confidence_after"].mean() - local["confidence_before"].mean()),
                }
            )
            rows.append(
                {
                    "metric": "semantic_drift_proxy_mean",
                    "group": group,
                    "method": method,
                    "coverage": np.nan,
                    "eps": eps,
                    "value": float(local["semantic_drift_proxy"].mean()),
                }
            )
    boot = pd.DataFrame(rows)
    if boot.empty:
        return boot
    keys = [col for col in ["metric", "group", "method", "eps", "coverage"] if col in boot.columns]
    return (
        boot.groupby(keys, dropna=False)["value"]
        .quantile([0.025, 0.5, 0.975])
        .unstack()
        .reset_index()
        .rename(columns={0.025: "ci_low", 0.5: "median", 0.975: "ci_high"})
    )


def write_docs(out_dir: Path, metadata: dict[str, object]) -> None:
    outputs = out_dir / "outputs"
    report = ["# Confidence Repair And Selective Abstention\n"]
    report.append("Objective test of confidence repair on overconfident-wrong cases and selective abstention policies.\n")
    report.append("## Setup\n")
    report.append("```json\n" + json.dumps(metadata, indent=2) + "\n```\n")
    for title, filename in [
        ("Group Counts", "group_counts.csv"),
        ("Calibration Metrics", "calibration_metrics.csv"),
        ("Repair Summary", "repair_summary.csv"),
        ("Selective Policy Benchmark", "selective_policy_benchmark.csv"),
        ("Selective Risk Curves", "selective_risk_curves.csv"),
        ("Route Score Summary", "route_score_summary.csv"),
        ("Bootstrap Confidence Intervals", "bootstrap_confidence_intervals.csv"),
    ]:
        path = outputs / filename
        if path.exists():
            df = pd.read_csv(path)
            report.append(f"## {title}\n")
            report.append("```text\n" + (df.head(60).to_string(index=False) if not df.empty else "(empty)") + "\n```\n")
    report.append("## Interpretation Guardrails\n")
    report.append(
        "- The intervention is local logit-lens hidden-state repair, not a full rollout benchmark.\n"
        "- Groups use absolute global confidence thresholds, not within-error medians.\n"
        "- The dataset is synthetic QA/reasoning; treat results as a scaling pilot, not deployment evidence.\n"
        "- A positive result requires improved selective risk/calibration without destroying correct answers; otherwise it is a failure mode.\n"
    )
    (out_dir / "reports" / "report.md").write_text("\n".join(report), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Confidence Repair And Selective Abstention\n\n"
        "Pilot application testing confidence repair on overconfident-wrong QA cases and answer/abstain policies.\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    ensure_dirs(args.out_dir)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    tokenizer, model, norm, lm_head = load_decoder(args)
    items = build_qa_items()
    base, records, answer_ids = build_records(args, tokenizer, model, norm, lm_head, items)
    scores, repair_raw = score_and_intervene(args, model, norm, lm_head, records)
    repair, repair_summary, calibration = summarize_repair(args, base, repair_raw)
    policy, risk_curves = selective_policy_tables(base, scores, repair)
    bootstrap_ci = bootstrap_confidence_intervals(
        base,
        repair,
        risk_curves,
        n_bootstrap=args.bootstrap,
        seed=args.seed + 17,
    )
    score_summary = scores.groupby(["route", "layer"], dropna=False).agg(n=("item_id", "size"), rho_mean=("rho", "mean"), rho_median=("rho", "median")).reset_index()
    group_counts = base.groupby(["outcome_group"], dropna=False).agg(n=("item_id", "size"), confidence_mean=("confidence", "mean"), accuracy=("correct", "mean")).reset_index()
    out = args.out_dir / "outputs"
    base.to_csv(out / "qa_base_records.csv", index=False)
    scores.to_csv(out / "route_scores.csv", index=False)
    score_summary.to_csv(out / "route_score_summary.csv", index=False)
    repair.to_csv(out / "repair_records.csv", index=False)
    repair_summary.to_csv(out / "repair_summary.csv", index=False)
    calibration.to_csv(out / "calibration_metrics.csv", index=False)
    policy.to_csv(out / "selective_policy_benchmark.csv", index=False)
    risk_curves.to_csv(out / "selective_risk_curves.csv", index=False)
    bootstrap_ci.to_csv(out / "bootstrap_confidence_intervals.csv", index=False)
    group_counts.to_csv(out / "group_counts.csv", index=False)
    metadata = {
        "status": "completed",
        "model": args.model,
        "device": str(next(model.parameters()).device),
        "torch_dtype": args.torch_dtype,
        "seed": args.seed,
        "n_items": len(items),
        "n_records": len(records),
        "answer_token_ids": dict(zip(ANSWER_LABELS, answer_ids)),
        "layers": args.layers,
        "pca_dim": args.pca_dim,
        "eps": args.eps,
        "high_confidence_threshold": args.high_confidence_threshold,
        "low_confidence_threshold": args.low_confidence_threshold,
        "max_items": args.max_items,
        "bootstrap": args.bootstrap,
    }
    (args.out_dir / "config" / "reproduce.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_docs(args.out_dir, metadata)


if __name__ == "__main__":
    main()
