from __future__ import annotations

import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from .rk2_week2 import (
    LABEL_NAMES,
    MODEL_NAME,
    MODEL_REVISION,
    SEED,
    calculate_metrics,
    load_articles,
    split_dataset,
    split_text,
)


EPOCHS = 4
BATCH_SIZE = 8
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
VALIDATION_SIZE = 0.20
MAX_TOKENS = 256


def set_reproducible_seed(seed: int = SEED) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))


def split_train_validation(
    articles: list[dict], train_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    labels = [articles[int(index)]["document_id"] for index in train_indices]
    fit_indices, validation_indices = train_test_split(
        train_indices,
        test_size=VALIDATION_SIZE,
        random_state=SEED,
        stratify=labels,
    )
    return np.sort(fit_indices), np.sort(validation_indices)


def build_chunk_records(
    articles: list[dict], indices: np.ndarray, label_to_id: dict[str, int]
) -> list[dict]:
    records = []
    for article_index in indices:
        article = articles[int(article_index)]
        chunks = split_text(article["text"])
        for chunk in chunks:
            records.append(
                {
                    "text": chunk,
                    "label": label_to_id[article["document_id"]],
                    "article_index": int(article_index),
                }
            )
    return records


class ChunkDataset:

    def __init__(self, records: list[dict]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]


def make_collator(tokenizer, *, max_tokens: int = MAX_TOKENS):
    import torch

    def collate(records: list[dict]) -> dict:
        encoded = tokenizer(
            [record["text"] for record in records],
            padding=True,
            truncation=True,
            max_length=max_tokens,
            return_tensors="pt",
        )
        encoded["labels"] = torch.tensor(
            [record["label"] for record in records], dtype=torch.long
        )
        encoded["article_indices"] = torch.tensor(
            [record["article_index"] for record in records], dtype=torch.long
        )
        return encoded

    return collate


def class_weights_from_records(records: list[dict], class_count: int):
    import torch

    counts = Counter(record["label"] for record in records)
    total = sum(counts.values())
    weights = [total / (class_count * counts[class_id]) for class_id in range(class_count)]
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_fine_tuned_model(
    model,
    tokenizer,
    articles: list[dict],
    indices: np.ndarray,
    label_to_id: dict[str, int],
    id_to_label: dict[int, str],
    loss_function,
    *,
    batch_size: int = BATCH_SIZE,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    import torch
    from torch.utils.data import DataLoader

    records = build_chunk_records(articles, indices, label_to_id)
    loader = DataLoader(
        ChunkDataset(records),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collator(tokenizer, max_tokens=max_tokens),
    )
    logits_by_article: defaultdict[int, list[np.ndarray]] = defaultdict(list)
    total_loss = 0.0
    total_examples = 0
    model.eval()
    with torch.no_grad():
        for batch in loader:
            article_indices = batch.pop("article_indices").numpy().tolist()
            labels = batch.pop("labels")
            logits = model(**batch).logits
            loss = loss_function(logits, labels)
            total_loss += float(loss.item()) * len(labels)
            total_examples += len(labels)
            for article_index, vector in zip(article_indices, logits.cpu().numpy()):
                logits_by_article[article_index].append(vector)

    true_labels = []
    predicted_labels = []
    confidences = []
    aggregated_logits = []
    for article_index in indices:
        article_index = int(article_index)
        mean_logits = np.mean(logits_by_article[article_index], axis=0)
        shifted = mean_logits - np.max(mean_logits)
        probabilities = np.exp(shifted) / np.exp(shifted).sum()
        predicted_id = int(np.argmax(probabilities))
        true_label = articles[article_index]["document_id"]
        true_labels.append(true_label)
        predicted_labels.append(id_to_label[predicted_id])
        confidences.append(float(probabilities[predicted_id]))
        aggregated_logits.append(mean_logits.tolist())

    return {
        "loss": round(total_loss / max(total_examples, 1), 6),
        "metrics": calculate_metrics(true_labels, predicted_labels),
        "true_labels": true_labels,
        "predicted_labels": predicted_labels,
        "confidences": confidences,
        "aggregated_logits": aggregated_logits,
        "chunk_count": len(records),
    }


def train_model(
    model,
    tokenizer,
    articles: list[dict],
    fit_indices: np.ndarray,
    validation_indices: np.ndarray,
    label_to_id: dict[str, int],
    id_to_label: dict[int, str],
    *,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    learning_rate: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    warmup_ratio: float = WARMUP_RATIO,
    max_tokens: int = MAX_TOKENS,
) -> tuple[list[dict], dict[str, object]]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    train_records = build_chunk_records(articles, fit_indices, label_to_id)
    loader = DataLoader(
        ChunkDataset(train_records),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
        collate_fn=make_collator(tokenizer, max_tokens=max_tokens),
    )
    class_weights = class_weights_from_records(train_records, len(label_to_id))
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    total_steps = epochs * len(loader)
    warmup_steps = max(1, int(total_steps * warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    history = []
    best_f1 = -1.0
    best_epoch = 0
    best_state = None
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in loader:
            batch.pop("article_indices")
            labels = batch.pop("labels")
            optimizer.zero_grad(set_to_none=True)
            logits = model(**batch).logits
            loss = loss_function(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            running_loss += float(loss.item()) * len(labels)
            seen += len(labels)

        validation = evaluate_fine_tuned_model(
            model,
            tokenizer,
            articles,
            validation_indices,
            label_to_id,
            id_to_label,
            loss_function,
            batch_size=batch_size,
            max_tokens=max_tokens,
        )
        row = {
            "epoch": epoch,
            "train_loss": round(running_loss / max(seen, 1), 6),
            "validation_loss": validation["loss"],
            **{f"validation_{key}": value for key, value in validation["metrics"].items()},
            "learning_rate_end": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        current_f1 = validation["metrics"]["f1_macro"]
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Не удалось сохранить лучшую эпоху")
    model.load_state_dict(best_state)
    return history, {
        "epoch": best_epoch,
        "validation_f1_macro": best_f1,
        "state": best_state,
        "loss_function": loss_function,
        "train_chunk_count": len(train_records),
        "warmup_steps": warmup_steps,
        "total_steps": total_steps,
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_learning_curves(history: list[dict], output_path: Path) -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="Train")
    axes[0].plot(
        epochs,
        [row["validation_loss"] for row in history],
        marker="o",
        label="Validation",
    )
    axes[0].set_title("Функция потерь")
    axes[0].set_xlabel("Эпоха")
    axes[0].grid(alpha=0.25)
    axes[0].legend()
    axes[1].plot(
        epochs,
        [row["validation_f1_macro"] for row in history],
        marker="o",
        color="#d97706",
    )
    axes[1].set_title("Validation macro-F1")
    axes[1].set_xlabel("Эпоха")
    axes[1].set_ylim(0, 1.03)
    axes[1].grid(alpha=0.25)
    fig.suptitle("Ход fine-tuning RuBERT", fontsize=15, weight="bold")
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_before_after_confusion(
    fine_tuned_result: dict,
    labels: list[str],
    output_path: Path,
) -> None:
    matrix = confusion_matrix(
        fine_tuned_result["true_labels"],
        fine_tuned_result["predicted_labels"],
        labels=labels,
        normalize="true",
    )
    short_names = ["Доступ", "ИИ", "Связь", "КБ", "Кодекс", "Перс. данные"]
    fig, axis = plt.subplots(figsize=(8.2, 7), constrained_layout=True)
    image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    axis.set_title("Fine-tuned RuBERT: нормированная матрица ошибок", fontsize=14, weight="bold")
    axis.set_xticks(range(len(labels)), short_names, rotation=35, ha="right")
    axis.set_yticks(range(len(labels)), short_names)
    axis.set_xlabel("Предсказанный класс")
    axis.set_ylabel("Истинный класс")
    for row in range(len(labels)):
        for column in range(len(labels)):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
            )
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

def run_week3_analysis(
    input_path: Path,
    week2_summary_path: Path,
    output_dir: Path,
) -> dict:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    set_reproducible_seed()
    output_dir.mkdir(parents=True, exist_ok=True)
    articles = load_articles(input_path)
    train_pool_indices, test_indices = split_dataset(articles)
    fit_indices, validation_indices = split_train_validation(articles, train_pool_indices)
    labels = sorted({article["document_id"] for article in articles})
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        revision=MODEL_REVISION,
        num_labels=len(labels),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    history, best = train_model(
        model,
        tokenizer,
        articles,
        fit_indices,
        validation_indices,
        label_to_id,
        id_to_label,
    )
    test_result = evaluate_fine_tuned_model(
        model,
        tokenizer,
        articles,
        test_indices,
        label_to_id,
        id_to_label,
        best["loss_function"],
    )

    model_dir = output_dir / "fine_tuned_model"
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    _write_csv(output_dir / "training_log.csv", history, list(history[0]))
    plot_learning_curves(history, output_dir / "learning_curves.png")
    plot_before_after_confusion(test_result, labels, output_dir / "confusion_matrix.png")

    week2_summary = json.loads(week2_summary_path.read_text(encoding="utf-8"))
    before_metrics = week2_summary["metrics"]["Frozen RuBERT + Logistic Regression"]
    after_metrics = test_result["metrics"]
    comparison_rows = [
        {"state": "before_frozen_bert", **before_metrics},
        {"state": "after_fine_tuning", **after_metrics},
    ]
    _write_csv(
        output_dir / "before_after_comparison.csv",
        comparison_rows,
        ["state", "accuracy", "precision_macro", "recall_macro", "f1_macro"],
    )

    prediction_rows = []
    for position, article_index in enumerate(test_indices):
        article = articles[int(article_index)]
        prediction_rows.append(
            {
                "article_id": article["article_id"],
                "document_title": article["document_title"],
                "true_label": test_result["true_labels"][position],
                "predicted_label": test_result["predicted_labels"][position],
                "correct": test_result["true_labels"][position]
                == test_result["predicted_labels"][position],
                "confidence": round(test_result["confidences"][position], 4),
                "preview": " ".join(article["text"].split())[:260],
            }
        )
    _write_csv(output_dir / "test_predictions.csv", prediction_rows, list(prediction_rows[0]))
    report = classification_report(
        test_result["true_labels"],
        test_result["predicted_labels"],
        labels=labels,
        output_dict=True,
        zero_division=0,
    )
    (output_dir / "classification_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    config = {
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "warmup_steps": best["warmup_steps"],
        "total_training_steps": best["total_steps"],
        "max_tokens": MAX_TOKENS,
        "seed": SEED,
        "class_weighting": "balanced by training chunk count",
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "week": 3,
        "task": "Fine-tuning RuBERT для классификации статей по исходному акту",
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "article_count": len(articles),
        "train_pool_count": len(train_pool_indices),
        "fit_count": len(fit_indices),
        "validation_count": len(validation_indices),
        "test_count": len(test_indices),
        "train_chunk_count": best["train_chunk_count"],
        "test_chunk_count": test_result["chunk_count"],
        "best_epoch": best["epoch"],
        "best_validation_f1_macro": round(float(best["validation_f1_macro"]), 4),
        "test_loss": test_result["loss"],
        "hyperparameters": config,
        "comparison": {
            "before_frozen_bert": before_metrics,
            "after_fine_tuning": after_metrics,
        },
        "f1_macro_delta": round(after_metrics["f1_macro"] - before_metrics["f1_macro"], 4),
        "test_isolation": "Тестовые статьи не использовались для обучения и выбора эпохи.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary

