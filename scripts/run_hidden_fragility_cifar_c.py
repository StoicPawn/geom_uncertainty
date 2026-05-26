from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "applications" / "06_hidden_fragility_cifar_c"
CIFAR10_MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
CIFAR10_STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
DEFAULT_CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "motion_blur",
    "defocus_blur",
    "brightness",
    "contrast",
    "fog",
    "frost",
    "elastic_transform",
    "jpeg_compression",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cifar10-dir", type=Path, default=ROOT / "data" / "cifar-10-batches-py")
    parser.add_argument("--cifar10c-dir", type=Path, default=ROOT / "data" / "CIFAR-10-C")
    parser.add_argument("--checkpoint", type=Path, default=OUT / "models" / "resnet18_cifar10.pt")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument("--mode", choices=["run", "write_skeleton"], default="run")
    parser.add_argument("--train-if-missing", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--keep-epoch-checkpoints", action="store_true")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--pca-dim", type=int, default=2)
    parser.add_argument("--confidence-quantile", type=float, default=0.70)
    parser.add_argument("--low-entropy-quantile", type=float, default=0.30)
    parser.add_argument("--high-margin-quantile", type=float, default=0.70)
    parser.add_argument("--rho-quantile", type=float, default=0.33)
    parser.add_argument("--match-caliper", type=float, default=1.50)
    parser.add_argument("--corruptions", default=",".join(DEFAULT_CORRUPTIONS))
    parser.add_argument("--severities", default="1,2,3,4,5")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=20260531)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def device_from_arg(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def ensure_dirs(out_dir: Path) -> None:
    for rel in ["config", "outputs", "reports", "models"]:
        (out_dir / rel).mkdir(parents=True, exist_ok=True)


def find_cifar10_root(path: Path) -> Path | None:
    candidates = [path, path / "cifar-10-batches-py"]
    for candidate in candidates:
        if (candidate / "test_batch").exists() and (candidate / "data_batch_1").exists():
            return candidate
    return None


def maybe_extract_cifar10(path: Path) -> Path | None:
    root = find_cifar10_root(path)
    if root is not None:
        return root
    archives = list(path.glob("cifar-10-python.tar.gz")) + list(path.parent.glob("cifar-10-python.tar.gz"))
    for archive in archives:
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(path=archive.parent)
        root = find_cifar10_root(archive.parent)
        if root is not None:
            return root
    return None


def find_cifar10c_root(path: Path) -> Path | None:
    candidates = [path, path / "CIFAR-10-C"]
    for candidate in candidates:
        has_corruption = any((candidate / f"{name}.npy").exists() for name in DEFAULT_CORRUPTIONS)
        if has_corruption and (candidate / "labels.npy").exists():
            return candidate
    return None


def load_cifar_batch(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("rb") as handle:
        data = pickle.load(handle, encoding="latin1")
    x = data["data"].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
    y = np.asarray(data.get("labels", data.get("fine_labels")), dtype=np.int64)
    return x, y


class ArrayDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, indices: np.ndarray | None = None):
        self.images = images
        self.labels = labels.astype(np.int64)
        self.indices = np.arange(len(labels), dtype=np.int64) if indices is None else indices.astype(np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image = torch.from_numpy(self.images[idx])
        image = (image - CIFAR10_MEAN) / CIFAR10_STD
        return image, torch.tensor(self.labels[idx], dtype=torch.long), torch.tensor(self.indices[idx], dtype=torch.long)


def load_cifar10(cifar10_dir: Path) -> tuple[ArrayDataset, ArrayDataset]:
    root = maybe_extract_cifar10(cifar10_dir)
    if root is None:
        raise FileNotFoundError(
            "CIFAR-10 python batches not found. Expected data/cifar-10-batches-py or a local "
            "cifar-10-python.tar.gz archive."
        )
    train_images, train_labels = [], []
    for i in range(1, 6):
        x, y = load_cifar_batch(root / f"data_batch_{i}")
        train_images.append(x)
        train_labels.append(y)
    x_train = np.concatenate(train_images, axis=0)
    y_train = np.concatenate(train_labels, axis=0)
    x_test, y_test = load_cifar_batch(root / "test_batch")
    return ArrayDataset(x_train, y_train), ArrayDataset(x_test, y_test)


def load_cifar10c(cifar10c_dir: Path, corruption: str, severity: int, clean_labels: np.ndarray) -> ArrayDataset:
    path = cifar10c_dir / f"{corruption}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing CIFAR-10-C corruption file: {path}")
    images = np.load(path, mmap_mode="r")
    start = (severity - 1) * 10000
    stop = severity * 10000
    if len(images) < stop:
        raise ValueError(f"{path} has {len(images)} rows, expected at least {stop}.")
    x = np.asarray(images[start:stop]).astype(np.float32)
    if x.shape[-1] == 3:
        x = np.transpose(x, (0, 3, 1, 2))
    if x.max() > 1.5:
        x = x / 255.0
    labels_path = cifar10c_dir / "labels.npy"
    if labels_path.exists():
        labels = np.load(labels_path)
        if len(labels) >= stop:
            labels = labels[start:stop]
        else:
            labels = labels[: len(x)]
    else:
        labels = clean_labels[: len(x)]
    return ArrayDataset(x, labels.astype(np.int64), indices=np.arange(len(labels), dtype=np.int64))


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut: nn.Module
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class ResNet18Cifar(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes, 1))
        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        out = F.relu(self.bn1(self.conv1(x)))
        l1 = self.layer1(out)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        pool = F.avg_pool2d(l4, 4).flatten(1)
        return pool, {"layer1": l1, "layer2": l2, "layer3": l3, "layer4": l4, "pool": pool}

    def forward(self, x: torch.Tensor, return_features: bool = False):
        pool, features = self.forward_features(x)
        logits = self.fc(pool)
        if return_features:
            return logits, features
        return logits


def train_model(
    model: nn.Module,
    train_ds: Dataset,
    checkpoint: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    max_train: int,
    resume_state: dict[str, object] | None = None,
    keep_epoch_checkpoints: bool = False,
) -> None:
    if max_train > 0:
        train_ds = Subset(train_ds, list(range(min(max_train, len(train_ds)))))
    loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[epochs // 2, int(epochs * 0.75)], gamma=0.1)
    start_epoch = 0
    if resume_state:
        start_epoch = int(resume_state.get("epoch", 0))
        if "optimizer" in resume_state:
            optimizer.load_state_dict(resume_state["optimizer"])
        if "scheduler" in resume_state:
            scheduler.load_state_dict(resume_state["scheduler"])
        print(f"resume_training epoch={start_epoch}/{epochs}", flush=True)
    model.train()
    for epoch in range(start_epoch, epochs):
        total_loss = 0.0
        total = 0
        correct = 0
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total += len(labels)
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
        scheduler.step()
        epoch_state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "epochs": epochs,
            "complete": epoch + 1 >= epochs,
            "train_loss": total_loss / max(total, 1),
            "train_accuracy": correct / max(total, 1),
        }
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(epoch_state, checkpoint)
        if keep_epoch_checkpoints:
            epoch_checkpoint = checkpoint.with_name(f"{checkpoint.stem}_epoch{epoch + 1:03d}{checkpoint.suffix}")
            torch.save(epoch_state, epoch_checkpoint)
        print(
            f"epoch={epoch + 1}/{epochs} loss={epoch_state['train_loss']:.4f} "
            f"acc={epoch_state['train_accuracy']:.4f} checkpoint={checkpoint}",
            flush=True,
        )
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "epoch": epochs,
            "epochs": epochs,
            "complete": True,
        },
        checkpoint,
    )


def load_or_train_model(args: argparse.Namespace, train_ds: Dataset, device: torch.device) -> ResNet18Cifar:
    model = ResNet18Cifar()
    if args.checkpoint.exists() and not args.force_retrain:
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state.get("model", state))
        complete = bool(state.get("complete", True))
        epoch = int(state.get("epoch", 0))
        if args.train_if_missing and not complete and epoch < args.epochs:
            train_model(
                model,
                train_ds,
                args.checkpoint,
                device,
                args.epochs,
                args.batch_size,
                args.lr,
                args.weight_decay,
                args.max_train,
                resume_state=state,
                keep_epoch_checkpoints=args.keep_epoch_checkpoints,
            )
    elif args.train_if_missing:
        train_model(
            model,
            train_ds,
            args.checkpoint,
            device,
            args.epochs,
            args.batch_size,
            args.lr,
            args.weight_decay,
            args.max_train,
            keep_epoch_checkpoints=args.keep_epoch_checkpoints,
        )
    else:
        raise FileNotFoundError(
            f"Missing checkpoint {args.checkpoint}. Pass --train-if-missing after placing CIFAR-10 locally, "
            "or provide --checkpoint."
        )
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def collect_embeddings(model: ResNet18Cifar, ds: Dataset, device: torch.device, batch_size: int, max_rows: int) -> np.ndarray:
    if max_rows > 0:
        ds = Subset(ds, list(range(min(max_rows, len(ds)))))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    rows = []
    for images, _, _ in loader:
        images = images.to(device)
        _, features = model(images, return_features=True)
        rows.append(features["pool"].detach().cpu().numpy())
    return np.concatenate(rows, axis=0)


def categorical_stats(logits: np.ndarray, true_label: int) -> dict[str, float | int]:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max()
    probs = np.exp(logits)
    probs = probs / probs.sum()
    pred = int(np.argmax(probs))
    sorted_probs = np.sort(probs)[::-1]
    entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    surprisal = -np.log(np.clip(probs, 1e-12, 1.0))
    varentropy = float(np.sum(probs * ((surprisal - entropy) ** 2)))
    loss = float(-math.log(max(float(probs[true_label]), 1e-12)))
    return {
        "pred": pred,
        "correct": int(pred == true_label),
        "confidence": float(sorted_probs[0]),
        "true_prob": float(probs[true_label]),
        "entropy": entropy,
        "varentropy": varentropy,
        "margin": float(sorted_probs[0] - sorted_probs[1]),
        "loss": loss,
    }


def fisher_sqrt(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs.astype(np.float64), 1e-12, 1.0)
    p = p / p.sum()
    fisher = np.diag(p) - np.outer(p, p)
    vals, vecs = np.linalg.eigh(fisher)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T


def projection_norm_sq(a: np.ndarray, w: np.ndarray) -> tuple[float, int]:
    if a.size == 0 or np.linalg.norm(a) < 1e-12:
        return 0.0, 0
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    tol = 1e-10 * max(a.shape) * (s[0] if len(s) else 0.0)
    rank = int(np.sum(s > tol))
    if rank == 0:
        return 0.0, 0
    q = u[:, :rank]
    projected = q @ (q.T @ w)
    return float(projected @ projected), rank


def rho_for_logits(logits: np.ndarray, basis: np.ndarray, classifier_weight: np.ndarray) -> tuple[float, float, float, int]:
    shifted = logits.astype(np.float64) - float(np.max(logits))
    probs = np.exp(shifted)
    probs = probs / probs.sum()
    entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    u = -np.log(np.clip(probs, 1e-12, 1.0)) - entropy
    f_sqrt = fisher_sqrt(probs)
    w = f_sqrt @ u
    total = float(w @ w)
    if total < 1e-12:
        return 0.0, total, 0.0, 0
    jb = classifier_weight @ basis
    a = f_sqrt @ jb
    access, rank = projection_norm_sq(a, w)
    return float(access / total), total, access, rank


def gradient_norms(logits: np.ndarray, true_label: int, classifier_weight: np.ndarray, basis: np.ndarray) -> dict[str, float]:
    shifted = logits.astype(np.float64) - float(np.max(logits))
    probs = np.exp(shifted)
    probs = probs / probs.sum()
    entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    u = -np.log(np.clip(probs, 1e-12, 1.0)) - entropy
    fisher = np.diag(probs) - np.outer(probs, probs)
    grad_h_logits = fisher @ u
    onehot = np.zeros_like(probs)
    onehot[true_label] = 1.0
    grad_loss_logits = probs - onehot
    grad_h_z = classifier_weight.T @ grad_h_logits
    grad_loss_z = classifier_weight.T @ grad_loss_logits
    return {
        "grad_entropy_norm": float(np.linalg.norm(grad_h_z)),
        "grad_loss_norm": float(np.linalg.norm(grad_loss_z)),
        "projected_grad_entropy_norm": float(np.linalg.norm(basis.T @ grad_h_z)),
        "projected_grad_loss_norm": float(np.linalg.norm(basis.T @ grad_loss_z)),
    }


@torch.no_grad()
def score_clean(
    model: ResNet18Cifar,
    ds: Dataset,
    basis: np.ndarray,
    device: torch.device,
    batch_size: int,
    max_test: int,
) -> pd.DataFrame:
    if max_test > 0:
        ds = Subset(ds, list(range(min(max_test, len(ds)))))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    classifier_weight = model.fc.weight.detach().cpu().numpy().astype(np.float64)
    rows = []
    for images, labels, indices in loader:
        images = images.to(device)
        logits_t, features = model(images, return_features=True)
        logits = logits_t.detach().cpu().numpy()
        z = features["pool"].detach().cpu().numpy()
        for i in range(len(labels)):
            y = int(labels[i].item())
            stats = categorical_stats(logits[i], y)
            rho, total_var, v_access, rank = rho_for_logits(logits[i], basis, classifier_weight)
            grads = gradient_norms(logits[i], y, classifier_weight, basis)
            rows.append(
                {
                    "index": int(indices[i].item()),
                    "true_label": y,
                    "true_class": CIFAR10_CLASSES[y],
                    **stats,
                    "rho": rho,
                    "v_total": total_var,
                    "v_access": v_access,
                    "v_inaccess": total_var - v_access,
                    "route_rank": rank,
                    "hidden_norm": float(np.linalg.norm(z[i])),
                    **grads,
                }
            )
    return pd.DataFrame(rows)


def fit_pca_basis(embeddings: np.ndarray, pca_dim: int) -> tuple[np.ndarray, dict[str, float]]:
    dim = embeddings.shape[1]
    k = min(pca_dim, dim, max(1, len(embeddings) - 1))
    pca = PCA(n_components=k, random_state=0)
    pca.fit(embeddings)
    basis = pca.components_.T.astype(np.float64)
    return basis, {
        "pca_dim_effective": float(k),
        "pca_explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }


def high_confidence_matched_pairs(
    clean: pd.DataFrame,
    confidence_quantile: float,
    low_entropy_quantile: float,
    high_margin_quantile: float,
    rho_quantile: float,
    caliper: float,
) -> pd.DataFrame:
    correct = clean[clean["correct"] == 1].copy()
    if correct.empty:
        return pd.DataFrame()
    conf_cut = float(correct["confidence"].quantile(confidence_quantile))
    entropy_cut = float(correct["entropy"].quantile(low_entropy_quantile))
    margin_cut = float(correct["margin"].quantile(high_margin_quantile))
    candidates = correct[
        (correct["confidence"] >= conf_cut)
        & (correct["entropy"] <= entropy_cut)
        & (correct["margin"] >= margin_cut)
    ].copy()
    if len(candidates) < 4:
        return pd.DataFrame()
    covariates = ["confidence", "entropy", "margin", "loss", "grad_entropy_norm", "projected_grad_entropy_norm"]
    rows = []
    pair_id = 0
    for cls, group in candidates.groupby("true_label"):
        if len(group) < 4:
            continue
        low_cut = float(group["rho"].quantile(rho_quantile))
        high_cut = float(group["rho"].quantile(1.0 - rho_quantile))
        low = group[group["rho"] <= low_cut].copy()
        high = group[group["rho"] >= high_cut].copy()
        if low.empty or high.empty:
            continue
        scaler = StandardScaler()
        scaler.fit(group[covariates].to_numpy(dtype=np.float64))
        low_x = scaler.transform(low[covariates].to_numpy(dtype=np.float64))
        high_x = scaler.transform(high[covariates].to_numpy(dtype=np.float64))
        used_high: set[int] = set()
        for i, (_, low_row) in enumerate(low.iterrows()):
            distances = np.linalg.norm(high_x - low_x[i], axis=1)
            order = np.argsort(distances)
            chosen = None
            chosen_distance = None
            for j in order:
                high_index = int(high.index[j])
                if high_index not in used_high:
                    chosen = high.iloc[j]
                    chosen_distance = float(distances[j])
                    used_high.add(high_index)
                    break
            if chosen is None:
                continue
            if chosen_distance is not None and chosen_distance > caliper:
                continue
            base = {
                "pair_id": pair_id,
                "true_label": int(cls),
                "true_class": CIFAR10_CLASSES[int(cls)],
                "match_distance": chosen_distance,
            }
            for prefix, row in [("low", low_row), ("high", chosen)]:
                for col in ["index", "pred", "confidence", "entropy", "margin", "loss", "rho", "grad_entropy_norm", "projected_grad_entropy_norm"]:
                    base[f"{prefix}_{col}"] = row[col]
            rows.append(base)
            pair_id += 1
    return pd.DataFrame(rows)


@torch.no_grad()
def score_corruption(
    model: ResNet18Cifar,
    ds: Dataset,
    selected: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    corruption: str,
    severity: int,
) -> pd.DataFrame:
    selected_by_index = {int(row["index"]): row for _, row in selected.iterrows()}
    wanted = sorted(selected_by_index.keys())
    subset = Subset(ds, wanted)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
    rows = []
    for images, labels, indices in loader:
        images = images.to(device)
        logits = model(images).detach().cpu().numpy()
        for i in range(len(labels)):
            idx = int(indices[i].item())
            clean_row = selected_by_index[idx]
            y = int(labels[i].item())
            stats = categorical_stats(logits[i], y)
            shifted = logits[i].astype(np.float64) - float(np.max(logits[i]))
            probs = np.exp(shifted)
            probs = probs / probs.sum()
            top10_jaccard = 1.0 if len(probs) <= 10 else np.nan
            rows.append(
                {
                    "index": idx,
                    "rho_group": clean_row["rho_group"],
                    "pair_id": int(clean_row["pair_id"]),
                    "true_label": y,
                    "true_class": CIFAR10_CLASSES[y],
                    "clean_pred": int(clean_row["pred"]),
                    "corrupt_pred": int(stats["pred"]),
                    "clean_correct": int(clean_row["correct"]),
                    "corrupt_correct": int(stats["correct"]),
                    "answer_flip": int(int(stats["pred"]) != int(clean_row["pred"])),
                    "correctness_loss": int(int(clean_row["correct"]) == 1 and int(stats["correct"]) == 0),
                    "clean_confidence": float(clean_row["confidence"]),
                    "corrupt_confidence": float(stats["confidence"]),
                    "clean_true_prob": float(clean_row["true_prob"]),
                    "corrupt_true_prob": float(stats["true_prob"]),
                    "true_prob_drop": float(clean_row["true_prob"]) - float(stats["true_prob"]),
                    "clean_entropy": float(clean_row["entropy"]),
                    "corrupt_entropy": float(stats["entropy"]),
                    "entropy_increase": float(stats["entropy"]) - float(clean_row["entropy"]),
                    "clean_margin": float(clean_row["margin"]),
                    "corrupt_margin": float(stats["margin"]),
                    "margin_drop": float(clean_row["margin"]) - float(stats["margin"]),
                    "top10_jaccard": top10_jaccard,
                    "corruption": corruption,
                    "severity": severity,
                    "rho": float(clean_row["rho"]),
                }
            )
    return pd.DataFrame(rows)


def build_selected_from_pairs(clean: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame()
    rows = []
    clean_by_index = clean.set_index("index")
    for pair in pairs.itertuples(index=False):
        for group, idx in [("low_rho", int(pair.low_index)), ("high_rho", int(pair.high_index))]:
            row = clean_by_index.loc[idx].to_dict()
            row["index"] = idx
            row["rho_group"] = group
            row["pair_id"] = int(pair.pair_id)
            row["match_distance"] = float(pair.match_distance)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_corruptions(records: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if records.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    summary = (
        records.groupby(["rho_group", "corruption", "severity"], dropna=False)
        .agg(
            n=("index", "size"),
            rho_mean=("rho", "mean"),
            clean_confidence_mean=("clean_confidence", "mean"),
            answer_flip_rate=("answer_flip", "mean"),
            corruption_error_rate=("correctness_loss", "mean"),
            true_prob_drop_mean=("true_prob_drop", "mean"),
            entropy_increase_mean=("entropy_increase", "mean"),
            margin_drop_mean=("margin_drop", "mean"),
            robust_accuracy=("corrupt_correct", "mean"),
        )
        .reset_index()
    )
    thresholds = []
    for (idx, group, corruption), g in records.groupby(["index", "rho_group", "corruption"]):
        flipped = g[g["answer_flip"] == 1]
        threshold = int(flipped["severity"].min()) if not flipped.empty else 6
        thresholds.append(
            {
                "index": idx,
                "rho_group": group,
                "corruption": corruption,
                "s_flip": threshold,
                "mean_true_prob_drop": float(g["true_prob_drop"].mean()),
                "mean_entropy_increase": float(g["entropy_increase"].mean()),
                "any_flip": int(threshold <= 5),
            }
        )
    threshold_df = pd.DataFrame(thresholds)
    pair_rows = []
    for (pair_id, corruption), g in records.groupby(["pair_id", "corruption"]):
        low = g[g["rho_group"] == "low_rho"]
        high = g[g["rho_group"] == "high_rho"]
        if low.empty or high.empty:
            continue
        pair_rows.append(
            {
                "pair_id": pair_id,
                "corruption": corruption,
                "low_flip_rate": float(low["answer_flip"].mean()),
                "high_flip_rate": float(high["answer_flip"].mean()),
                "low_minus_high_flip_rate": float(low["answer_flip"].mean() - high["answer_flip"].mean()),
                "low_true_prob_drop": float(low["true_prob_drop"].mean()),
                "high_true_prob_drop": float(high["true_prob_drop"].mean()),
                "low_minus_high_true_prob_drop": float(low["true_prob_drop"].mean() - high["true_prob_drop"].mean()),
                "low_entropy_increase": float(low["entropy_increase"].mean()),
                "high_entropy_increase": float(high["entropy_increase"].mean()),
                "low_minus_high_entropy_increase": float(low["entropy_increase"].mean() - high["entropy_increase"].mean()),
                "low_robust_accuracy": float(low["corrupt_correct"].mean()),
                "high_robust_accuracy": float(high["corrupt_correct"].mean()),
                "high_minus_low_robust_accuracy": float(high["corrupt_correct"].mean() - low["corrupt_correct"].mean()),
            }
        )
    contrasts = pd.DataFrame(pair_rows)
    return summary, threshold_df, contrasts


def predictor_benchmark(clean: pd.DataFrame, records: pd.DataFrame) -> pd.DataFrame:
    if clean.empty or records.empty:
        return pd.DataFrame()
    outcomes = (
        records.groupby("index")
        .agg(
            any_flip=("answer_flip", "max"),
            mean_true_prob_drop=("true_prob_drop", "mean"),
            mean_entropy_increase=("entropy_increase", "mean"),
            corruption_error=("correctness_loss", "max"),
        )
        .reset_index()
    )
    df = clean.merge(outcomes, on="index", how="inner")
    predictors = ["rho", "confidence", "entropy", "margin", "loss", "grad_entropy_norm", "projected_grad_entropy_norm", "hidden_norm"]
    df = df.dropna(subset=predictors + ["any_flip", "mean_true_prob_drop", "corruption_error"])
    if len(df) < 20:
        return pd.DataFrame()
    x = df[predictors].to_numpy(dtype=np.float64)
    x = StandardScaler().fit_transform(x)
    rows = []
    for outcome, kind in [("any_flip", "binary"), ("corruption_error", "binary"), ("mean_true_prob_drop", "continuous")]:
        y = df[outcome].to_numpy(dtype=np.float64)
        if kind == "binary" and len(np.unique(y)) < 2:
            continue
        for name, cols in [
            ("scalar_baseline", ["confidence", "entropy", "margin", "loss"]),
            ("gradient_baseline", ["confidence", "entropy", "margin", "loss", "grad_entropy_norm", "projected_grad_entropy_norm"]),
            ("gradient_plus_rho", predictors),
            ("rho_only", ["rho"]),
        ]:
            col_idx = [predictors.index(c) for c in cols]
            xx = x[:, col_idx]
            if kind == "binary":
                model = LogisticRegression(max_iter=2000, class_weight="balanced")
                model.fit(xx, y)
                scores = model.predict_proba(xx)[:, 1]
                auroc = float(roc_auc_score(y, scores))
                auprc = float(average_precision_score(y, scores))
                rho_coef = float(model.coef_[0][cols.index("rho")]) if "rho" in cols else np.nan
                rows.append({"outcome": outcome, "model": name, "metric": "auroc", "value": auroc, "rho_coef": rho_coef, "n": len(df)})
                rows.append({"outcome": outcome, "model": name, "metric": "auprc", "value": auprc, "rho_coef": rho_coef, "n": len(df)})
            else:
                model = LinearRegression()
                model.fit(xx, y)
                pred = model.predict(xx)
                r2 = float(model.score(xx, y))
                sp = float(spearmanr(pred, y).correlation)
                rho_coef = float(model.coef_[cols.index("rho")]) if "rho" in cols else np.nan
                rows.append({"outcome": outcome, "model": name, "metric": "r2", "value": r2, "rho_coef": rho_coef, "n": len(df)})
                rows.append({"outcome": outcome, "model": name, "metric": "spearman", "value": sp, "rho_coef": rho_coef, "n": len(df)})
    return pd.DataFrame(rows)


def write_skeleton(out_dir: Path, args: argparse.Namespace, reason: str) -> None:
    ensure_dirs(out_dir)
    config = {
        "application": "hidden fragility in confident predictions",
        "status": "not_run",
        "reason": reason,
        "command": (
            "python scripts\\run_hidden_fragility_cifar_c.py "
            "--cifar10-dir data\\cifar-10-batches-py --cifar10c-dir data\\CIFAR-10-C "
            "--checkpoint applications\\06_hidden_fragility_cifar_c\\models\\resnet18_cifar10.pt "
            "--train-if-missing --epochs 40 --pca-dim 2 --confidence-quantile 0.70 --seed 20260531"
        ),
        "dataset": "CIFAR-10 clean test plus CIFAR-10-C corruptions, severities 1-5",
        "model": "ResNet-18 for CIFAR-10, implemented locally without torchvision",
        "route": "penultimate embedding PCA subspace",
        "matching": "correct, high-confidence, low-entropy, high-margin; paired within class on scalar and gradient covariates",
        "confidence_quantile": args.confidence_quantile,
        "low_entropy_quantile": args.low_entropy_quantile,
        "high_margin_quantile": args.high_margin_quantile,
        "rho_quantile": args.rho_quantile,
        "seed": args.seed,
    }
    (out_dir / "config" / "reproduce.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Hidden Fragility In Confident Predictions\n\n"
        "Separate vision-domain application for CIFAR-10 -> CIFAR-10-C.\n\n"
        "## Claim\n\n"
        "Among equally confident and correct predictions, accessible-varentropy should distinguish robust from brittle examples under label-preserving corruptions.\n\n"
        "## Status\n\n"
        f"Not executed in this environment: {reason}\n\n"
        "## Reproduce\n\n"
        "Place CIFAR-10 python batches under `data/cifar-10-batches-py` and CIFAR-10-C `.npy` files under `data/CIFAR-10-C`, then run:\n\n"
        "```powershell\n"
        "python scripts\\run_hidden_fragility_cifar_c.py --cifar10-dir data\\cifar-10-batches-py --cifar10c-dir data\\CIFAR-10-C --train-if-missing --epochs 40 --pca-dim 2 --confidence-quantile 0.70 --seed 20260531\n"
        "```\n\n"
        "The outputs are written to `outputs/` and summarized in `reports/report.md`.\n",
        encoding="utf-8",
    )
    (out_dir / "reports" / "report.md").write_text(
        "# Hidden Fragility In Confident Predictions\n\n"
        "This application is configured but not executed in the current local environment.\n\n"
        f"Reason: {reason}\n\n"
        "Expected outputs after execution:\n\n"
        "- `clean_scores.csv`\n"
        "- `matched_high_confidence_pairs.csv`\n"
        "- `corruption_records.csv`\n"
        "- `flip_thresholds.csv`\n"
        "- `corruption_group_summary.csv`\n"
        "- `matched_fragility_contrasts.csv`\n"
        "- `predictor_benchmark.csv`\n",
        encoding="utf-8",
    )


def write_report(out_dir: Path, metadata: dict[str, object]) -> None:
    outputs = out_dir / "outputs"
    parts = ["# Hidden Fragility In Confident Predictions\n"]
    parts.append("Vision-domain application using CIFAR-10 clean predictions and CIFAR-10-C label-preserving corruptions.\n")
    parts.append("## Setup\n")
    parts.append("```json\n" + json.dumps(metadata, indent=2) + "\n```\n")
    for title, filename in [
        ("Clean High-Confidence Matching", "matched_high_confidence_pairs.csv"),
        ("Corruption Group Summary", "corruption_group_summary.csv"),
        ("Matched Fragility Contrasts", "matched_fragility_contrasts.csv"),
        ("Predictor Benchmark", "predictor_benchmark.csv"),
    ]:
        path = outputs / filename
        if not path.exists():
            continue
        df = pd.read_csv(path)
        parts.append(f"## {title}\n")
        if df.empty:
            parts.append("```text\n(empty)\n```\n")
        else:
            parts.append("```text\n" + df.head(30).to_string(index=False) + "\n```\n")
    (out_dir / "reports" / "report.md").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    ensure_dirs(args.out_dir)
    if args.mode == "write_skeleton":
        write_skeleton(args.out_dir, args, "skeleton requested")
        return

    cifar10_root = maybe_extract_cifar10(args.cifar10_dir)
    cifar10c_root = find_cifar10c_root(args.cifar10c_dir)
    missing = []
    if cifar10_root is None:
        missing.append(f"CIFAR-10 python batches at {args.cifar10_dir}")
    if cifar10c_root is None:
        missing.append(f"CIFAR-10-C directory at {args.cifar10c_dir}")
    if missing:
        write_skeleton(args.out_dir, args, "; ".join(missing))
        raise FileNotFoundError("; ".join(missing))

    device = device_from_arg(args.device)
    train_ds, test_ds = load_cifar10(cifar10_root)
    if args.max_test > 0:
        test_ds = Subset(test_ds, list(range(min(args.max_test, len(test_ds)))))
    model = load_or_train_model(args, train_ds, device)

    embeddings = collect_embeddings(model, train_ds, device, args.batch_size, args.max_train or 20000)
    basis, pca_meta = fit_pca_basis(embeddings, args.pca_dim)
    clean = score_clean(model, test_ds, basis, device, args.batch_size, args.max_test)
    pairs = high_confidence_matched_pairs(
        clean,
        args.confidence_quantile,
        args.low_entropy_quantile,
        args.high_margin_quantile,
        args.rho_quantile,
        args.match_caliper,
    )
    selected = build_selected_from_pairs(clean, pairs)

    out = args.out_dir / "outputs"
    clean.to_csv(out / "clean_scores.csv", index=False)
    pairs.to_csv(out / "matched_high_confidence_pairs.csv", index=False)
    selected.to_csv(out / "matched_selected_clean_rows.csv", index=False)

    records = []
    labels = np.asarray([test_ds[i][1].item() for i in range(len(test_ds))], dtype=np.int64)
    corruptions = [c.strip() for c in args.corruptions.split(",") if c.strip()]
    severities = [int(s.strip()) for s in args.severities.split(",") if s.strip()]
    if not selected.empty:
        for corruption in corruptions:
            for severity in severities:
                ds_c = load_cifar10c(cifar10c_root, corruption, severity, labels)
                if args.max_test > 0:
                    ds_c = Subset(ds_c, list(range(min(args.max_test, len(ds_c)))))
                records.append(score_corruption(model, ds_c, selected, device, args.batch_size, corruption, severity))
    records_df = pd.concat(records, ignore_index=True) if records else pd.DataFrame()
    summary, thresholds, contrasts = summarize_corruptions(records_df)
    predictors = predictor_benchmark(clean, records_df)
    records_df.to_csv(out / "corruption_records.csv", index=False)
    summary.to_csv(out / "corruption_group_summary.csv", index=False)
    thresholds.to_csv(out / "flip_thresholds.csv", index=False)
    contrasts.to_csv(out / "matched_fragility_contrasts.csv", index=False)
    predictors.to_csv(out / "predictor_benchmark.csv", index=False)

    metadata = {
        "status": "completed",
        "seed": args.seed,
        "device": str(device),
        "checkpoint": str(args.checkpoint),
        "cifar10_dir": str(cifar10_root),
        "cifar10c_dir": str(cifar10c_root),
        "pca_dim_requested": args.pca_dim,
        **pca_meta,
        "confidence_quantile": args.confidence_quantile,
        "low_entropy_quantile": args.low_entropy_quantile,
        "high_margin_quantile": args.high_margin_quantile,
        "rho_quantile": args.rho_quantile,
        "corruptions": corruptions,
        "severities": severities,
        "n_clean_scored": int(len(clean)),
        "n_matched_pairs": int(len(pairs)),
        "n_corruption_records": int(len(records_df)),
    }
    (args.out_dir / "config" / "reproduce.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    write_report(args.out_dir, metadata)


if __name__ == "__main__":
    main()
