from __future__ import annotations

import copy
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


LOG_FEATURES = {
    "cdr_events_total",
    "cdr_direction_min_count",
    "cdr_direction_max_count",
    "cdr_total_duration_seconds",
    "cdr_mean_duration_seconds",
    "cdr_max_duration_seconds",
    "cdr_duration_std_seconds",
    "cdr_active_days",
    "cdr_activity_span_days",
    "cdr_events_per_active_day",
    "cdr_degree_min",
    "cdr_degree_max",
    "cdr_common_contacts",
    "ipdr_session_count_min",
    "ipdr_session_count_max",
    "ipdr_total_mb_min",
    "ipdr_total_mb_max",
    "ipdr_active_days_min",
    "ipdr_active_days_max",
}


class RelationshipMLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: tuple[int, ...] = (64, 32), dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_size
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(dropout)])
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values).squeeze(1)


def choose_device(requested: str) -> torch.device:
    requested = requested.casefold()
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch cannot access an NVIDIA GPU.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _stratified_split(labels: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    splits: dict[str, list[int]] = {"train": [], "validation": [], "test": []}
    for class_value in (0, 1):
        indices = np.flatnonzero(labels == class_value)
        if len(indices) < 3:
            raise ValueError(
                "Base truth needs at least 3 positive and 3 negative pairs; "
                f"class {class_value} has only {len(indices)}."
            )
        rng.shuffle(indices)
        test_count = max(1, int(round(len(indices) * 0.20)))
        validation_count = max(1, int(round(len(indices) * 0.20)))
        while test_count + validation_count > len(indices) - 1:
            if validation_count > 1:
                validation_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                break
        splits["test"].extend(indices[:test_count].tolist())
        splits["validation"].extend(indices[test_count : test_count + validation_count].tolist())
        splits["train"].extend(indices[test_count + validation_count :].tolist())
    result: dict[str, np.ndarray] = {}
    for name, values in splits.items():
        array = np.asarray(values, dtype=np.int64)
        rng.shuffle(array)
        result[name] = array
    return result


def _transform_matrix(
    frame: pd.DataFrame,
    feature_names: list[str],
    means: np.ndarray | None = None,
    scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = frame[feature_names].to_numpy(dtype=np.float64, copy=True)
    matrix[~np.isfinite(matrix)] = 0.0
    for index, feature in enumerate(feature_names):
        if feature in LOG_FEATURES:
            matrix[:, index] = np.log1p(np.clip(matrix[:, index], 0.0, None))
    if means is None:
        means = matrix.mean(axis=0)
    if scales is None:
        scales = matrix.std(axis=0)
        scales[scales < 1e-8] = 1.0
    return ((matrix - means) / scales).astype(np.float32), means, scales


def _metric_dict(labels: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, object]:
    predictions = (probabilities >= threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    result: dict[str, object] = {
        "rows": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    if len(np.unique(labels)) == 2:
        result["roc_auc"] = float(roc_auc_score(labels, probabilities))
        result["average_precision"] = float(average_precision_score(labels, probabilities))
    return result


def _best_threshold(labels: np.ndarray, probabilities: np.ndarray) -> float:
    candidates = np.linspace(0.05, 0.95, 181)
    scored = [
        (float(f1_score(labels, probabilities >= threshold, zero_division=0)), -abs(threshold - 0.5), threshold)
        for threshold in candidates
    ]
    return float(max(scored)[2])


def _probabilities(model: nn.Module, matrix: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(matrix).to(device)
        return torch.sigmoid(model(tensor)).detach().cpu().numpy().astype(np.float64)


def train_neural_network(
    features: pd.DataFrame,
    labels: np.ndarray,
    feature_names: list[str],
    epochs: int = 250,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 42,
    device_name: str = "auto",
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, object], dict[str, object], np.ndarray, np.ndarray, pd.DataFrame]:
    progress = progress or (lambda message: None)
    labels = np.asarray(labels, dtype=np.int64)
    if len(labels) != len(features):
        raise ValueError("Feature and label row counts do not match.")
    if len(labels) < 10:
        raise ValueError("Provide at least 10 labeled pairs. Hundreds or more are strongly recommended.")
    if set(np.unique(labels)) != {0, 1}:
        raise ValueError("Base truth must contain both positive and negative labels.")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive.")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = choose_device(device_name)
    progress(f"Neural network device: {device}")

    splits = _stratified_split(labels, seed)
    raw_matrix, _, _ = _transform_matrix(features, feature_names)
    del raw_matrix
    training_raw = features.iloc[splits["train"]]
    _, means, scales = _transform_matrix(training_raw, feature_names)
    matrix, _, _ = _transform_matrix(features, feature_names, means, scales)

    train_x = torch.from_numpy(matrix[splits["train"]])
    train_y = torch.from_numpy(labels[splits["train"]].astype(np.float32))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=min(batch_size, len(train_x)),
        shuffle=True,
        generator=generator,
    )

    hidden_sizes = (64, 32)
    dropout = 0.20
    model = RelationshipMLP(len(feature_names), hidden_sizes, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_labels = labels[splits["train"]]
    positive_count = max(int((train_labels == 1).sum()), 1)
    negative_count = max(int((train_labels == 0).sum()), 1)
    positive_weight = torch.tensor([negative_count / positive_count], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)

    validation_x = torch.from_numpy(matrix[splits["validation"]]).to(device)
    validation_y = torch.from_numpy(labels[splits["validation"]].astype(np.float32)).to(device)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = math.inf
    best_epoch = 0
    patience = min(35, max(12, epochs // 6))
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch_x)
            total_rows += len(batch_x)
        model.eval()
        with torch.no_grad():
            validation_loss = float(criterion(model(validation_x), validation_y).detach().cpu())
        training_loss = total_loss / max(total_rows, 1)
        history.append({"epoch": epoch, "training_loss": training_loss, "validation_loss": validation_loss})
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if epoch == 1 or epoch % 10 == 0:
            progress(
                f"Epoch {epoch}/{epochs}: train loss={training_loss:.4f}, "
                f"validation loss={validation_loss:.4f}"
            )
        if stale_epochs >= patience:
            progress(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    model.load_state_dict(best_state)
    all_probabilities = _probabilities(model, matrix, device)
    threshold = _best_threshold(labels[splits["validation"]], all_probabilities[splits["validation"]])
    all_predictions = (all_probabilities >= threshold).astype(np.int64)
    metrics = {
        "threshold": threshold,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "device": str(device),
        "class_counts": {
            "negative": int((labels == 0).sum()),
            "positive": int((labels == 1).sum()),
        },
        "train": _metric_dict(labels[splits["train"]], all_probabilities[splits["train"]], threshold),
        "validation": _metric_dict(
            labels[splits["validation"]], all_probabilities[splits["validation"]], threshold
        ),
        "test": _metric_dict(labels[splits["test"]], all_probabilities[splits["test"]], threshold),
        "warning": (
            "This is a statistical prediction learned from supplied labels, not proof of a personal relationship."
        ),
    }

    cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    artifact: dict[str, object] = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "feature_names": feature_names,
        "log_features": sorted(LOG_FEATURES),
        "scaler_mean": means.tolist(),
        "scaler_scale": scales.tolist(),
        "threshold": threshold,
        "input_size": len(feature_names),
        "hidden_sizes": list(hidden_sizes),
        "dropout": dropout,
        "state_dict": cpu_state,
        "training": {
            "seed": seed,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "epochs_requested": epochs,
            "best_epoch": best_epoch,
        },
    }

    split_names = np.full(len(labels), "", dtype=object)
    for split_name, indices in splits.items():
        split_names[indices] = split_name

    first_layer = model.network[0]
    assert isinstance(first_layer, nn.Linear)
    importances = first_layer.weight.detach().abs().mean(dim=0).cpu().numpy()
    importance_sum = float(importances.sum())
    if importance_sum:
        importances = importances / importance_sum
    importance_frame = pd.DataFrame(
        {"feature": feature_names, "relative_input_weight": importances}
    ).sort_values("relative_input_weight", ascending=False, ignore_index=True)
    return artifact, metrics, all_probabilities, split_names, importance_frame


def save_artifact(artifact: dict[str, object], path: str | Path) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, target)


def load_artifact(path: str | Path) -> dict[str, object]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Model file not found: {source}")
    try:
        artifact = torch.load(source, map_location="cpu", weights_only=True)
    except TypeError:
        artifact = torch.load(source, map_location="cpu")
    if not isinstance(artifact, dict) or artifact.get("format_version") != 1:
        raise ValueError("Unsupported or invalid model artifact.")
    return artifact


def predict_with_artifact(
    artifact: dict[str, object], features: pd.DataFrame, device_name: str = "auto"
) -> tuple[np.ndarray, np.ndarray]:
    feature_names = [str(value) for value in artifact["feature_names"]]
    missing = [feature for feature in feature_names if feature not in features.columns]
    if missing:
        raise ValueError(f"Prediction features are missing: {missing}")
    means = np.asarray(artifact["scaler_mean"], dtype=np.float64)
    scales = np.asarray(artifact["scaler_scale"], dtype=np.float64)
    matrix, _, _ = _transform_matrix(features, feature_names, means, scales)
    model = RelationshipMLP(
        int(artifact["input_size"]),
        tuple(int(value) for value in artifact["hidden_sizes"]),
        float(artifact["dropout"]),
    )
    model.load_state_dict(artifact["state_dict"])
    device = choose_device(device_name)
    model.to(device)
    probabilities = _probabilities(model, matrix, device)
    predictions = (probabilities >= float(artifact["threshold"])).astype(np.int64)
    return probabilities, predictions


def write_json(value: dict[str, object], path: str | Path) -> None:
    target = Path(path).expanduser().resolve()
    target.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")

