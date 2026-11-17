"""Пятая неделя второго РК: инференс fine-tuned RuBERT для веб-интерфейса."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .rk2_week2 import LABEL_NAMES, MAX_TOKENS, split_dataset, split_text


MIN_INPUT_CHARACTERS = 20


def validate_input_text(text: str) -> str:
    """Нормализовать пользовательский текст и отклонить пустой ввод."""
    normalized = " ".join(text.split())
    if len(normalized) < MIN_INPUT_CHARACTERS:
        raise ValueError(
            f"Введите осмысленный фрагмент длиной не менее {MIN_INPUT_CHARACTERS} символов."
        )
    return normalized


def load_model_bundle(model_dir: Path) -> tuple[Any, Any]:
    """Загрузить локальную модель и токенизатор без обращения к интернету."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not (model_dir / "model.safetensors").exists():
        raise FileNotFoundError(f"Не найдена обученная модель: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True
    )
    model.eval()
    return tokenizer, model


def confidence_level(confidence: float) -> str:
    """Перевести числовую уверенность в понятную учебную подпись."""
    if confidence >= 0.75:
        return "высокая"
    if confidence >= 0.45:
        return "средняя"
    return "низкая"


def predict_text(
    text: str,
    tokenizer: Any,
    model: Any,
    *,
    max_tokens: int = MAX_TOKENS,
) -> dict:
    """Классифицировать текст, усреднив логиты всех его фрагментов."""
    import torch

    normalized = validate_input_text(text)
    chunks = split_text(normalized)
    encoded = tokenizer(
        chunks,
        padding=True,
        truncation=True,
        max_length=max_tokens,
        return_tensors="pt",
    )
    with torch.no_grad():
        chunk_logits = model(**encoded).logits
        article_logits = chunk_logits.mean(dim=0)
        probabilities = torch.softmax(article_logits, dim=0).cpu().numpy()

    id_to_label = {
        int(class_id): label for class_id, label in model.config.id2label.items()
    }
    ranked_ids = np.argsort(probabilities)[::-1]
    ranked = [
        {
            "label": id_to_label[int(class_id)],
            "label_ru": LABEL_NAMES.get(
                id_to_label[int(class_id)], id_to_label[int(class_id)]
            ),
            "probability": round(float(probabilities[int(class_id)]), 6),
        }
        for class_id in ranked_ids
    ]
    winner = ranked[0]
    return {
        "predicted_label": winner["label"],
        "predicted_label_ru": winner["label_ru"],
        "confidence": winner["probability"],
        "confidence_level": confidence_level(winner["probability"]),
        "probabilities": ranked,
        "chunk_count": len(chunks),
        "word_count": len(normalized.split()),
    }


def select_control_examples(articles: list[dict], *, count: int = 5) -> list[dict]:
    """Выбрать по одному воспроизводимому test-примеру из разных классов."""
    if count < 1:
        raise ValueError("Нужно выбрать хотя бы один контрольный пример")
    _, test_indices = split_dataset(articles)
    selected = []
    seen_labels: set[str] = set()
    for index in test_indices:
        article = articles[int(index)]
        label = article["document_id"]
        if label in seen_labels:
            continue
        selected.append(
            {
                "sample_id": len(selected) + 1,
                "article_id": article["article_id"],
                "article_title": article.get("article_title", ""),
                "expected_label": label,
                "expected_label_ru": LABEL_NAMES.get(label, label),
                "text": article["text"],
                "word_count": article.get(
                    "clean_word_count", len(article["text"].split())
                ),
                "source": "held-out test; пример не участвовал в обучении",
            }
        )
        seen_labels.add(label)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Удалось выбрать только {len(selected)} разных классов")
    return selected


def load_control_examples(path: Path) -> list[dict]:
    """Прочитать подготовленные контрольные тексты для UI."""
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Файл контрольных примеров должен содержать список")
    return payload
