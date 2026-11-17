from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


SEED = 42
MODEL_NAME = "cointegrated/rubert-tiny2"
MODEL_REVISION = "776805cd8327870dc89ab911826f5e01462edb20"
MIN_WORDS = 20
TEST_SIZE = 0.25
MAX_WORDS_PER_CHUNK = 180
MAX_TOKENS = 256
LABEL_NAMES = {
    "access_to_information": "Доступ к информации",
    "artificial_intelligence": "Искусственный интеллект",
    "communications": "Связь",
    "cybersecurity": "Кибербезопасность",
    "digital_code": "Цифровой кодекс",
    "personal_data": "Персональные данные",
}


def load_articles(path: Path, *, min_words: int = MIN_WORDS) -> list[dict]:
    articles = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    filtered = [article for article in articles if article["clean_word_count"] >= min_words]
    if not filtered:
        raise ValueError("После фильтрации не осталось статей")
    return filtered


def split_dataset(
    articles: list[dict], *, test_size: float = TEST_SIZE, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(len(articles))
    labels = [article["document_id"] for article in articles]
    train_indices, test_indices = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )
    return np.sort(train_indices), np.sort(test_indices)


def split_text(text: str, *, max_words: int = MAX_WORDS_PER_CHUNK) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?;])\s+", text) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    word_count = 0
    for sentence in sentences:
        sentence_words = sentence.split()
        if len(sentence_words) > max_words:
            if current:
                chunks.append(" ".join(current))
                current = []
                word_count = 0
            chunks.extend(
                " ".join(sentence_words[start : start + max_words])
                for start in range(0, len(sentence_words), max_words)
            )
            continue
        if current and word_count + len(sentence_words) > max_words:
            chunks.append(" ".join(current))
            current = []
            word_count = 0
        current.append(sentence)
        word_count += len(sentence_words)
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


def aggregate_chunk_vectors(
    chunk_vectors: np.ndarray, owners: list[int], article_count: int
) -> np.ndarray:
    article_vectors = []
    owners_array = np.asarray(owners)
    for article_index in range(article_count):
        owned = chunk_vectors[owners_array == article_index]
        if len(owned) == 0:
            raise ValueError(f"Для статьи {article_index} нет фрагментов")
        vector = owned.mean(axis=0)
        norm = np.linalg.norm(vector)
        article_vectors.append(vector / norm if norm else vector)
    return np.asarray(article_vectors, dtype=np.float32)


def encode_with_frozen_bert(
    texts: list[str], tokenizer, model, *, batch_size: int = 16
) -> tuple[np.ndarray, list[int]]:
    import torch
    import torch.nn.functional as functional

    chunks: list[str] = []
    owners: list[int] = []
    chunk_counts: list[int] = []
    for article_index, text in enumerate(texts):
        article_chunks = split_text(text)
        chunks.extend(article_chunks)
        owners.extend([article_index] * len(article_chunks))
        chunk_counts.append(len(article_chunks))

    model.eval()
    batches = []
    for start in range(0, len(chunks), batch_size):
        encoded = tokenizer(
            chunks[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=MAX_TOKENS,
            return_tensors="pt",
        )
        with torch.no_grad():
            output = model(**encoded).last_hidden_state[:, 0, :]
            output = functional.normalize(output, p=2, dim=1)
        batches.append(output.cpu().numpy())
    chunk_vectors = np.vstack(batches).astype(np.float32)
    return aggregate_chunk_vectors(chunk_vectors, owners, len(texts)), chunk_counts


def dataset_fingerprint(articles: list[dict]) -> str:
    digest = hashlib.sha256()
    for article in articles:
        digest.update(article["article_id"].encode("utf-8"))
        digest.update(article["text"].encode("utf-8"))
    return digest.hexdigest()


def load_or_create_bert_embeddings(
    articles: list[dict], output_dir: Path
) -> tuple[np.ndarray, list[int], bool]:
    embeddings_path = output_dir / "bert_embeddings.npy"
    metadata_path = output_dir / "bert_embedding_metadata.json"
    fingerprint = dataset_fingerprint(articles)
    if embeddings_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("model_name") == MODEL_NAME
            and metadata.get("model_revision") == MODEL_REVISION
            and metadata.get("dataset_fingerprint") == fingerprint
            and metadata.get("article_count") == len(articles)
        ):
            return np.load(embeddings_path), metadata["chunk_counts"], True

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    model = AutoModel.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    texts = [article["text"] for article in articles]
    embeddings, chunk_counts = encode_with_frozen_bert(texts, tokenizer, model)
    np.save(embeddings_path, embeddings)
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": MODEL_NAME,
                "model_revision": MODEL_REVISION,
                "dataset_fingerprint": fingerprint,
                "article_count": len(articles),
                "embedding_dimension": int(embeddings.shape[1]),
                "chunk_counts": chunk_counts,
                "max_words_per_chunk": MAX_WORDS_PER_CHUNK,
                "max_tokens": MAX_TOKENS,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return embeddings, chunk_counts, False


def build_classifier() -> LogisticRegression:
    return LogisticRegression(
        C=4.0,
        class_weight="balanced",
        max_iter=3000,
        random_state=SEED,
    )


def calculate_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(
            float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "recall_macro": round(
            float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4
        ),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
    }


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_confusion_matrices(
    y_true: list[str],
    baseline_predictions: list[str],
    bert_predictions: list[str],
    labels: list[str],
    output_path: Path,
) -> None:
    matrices = [
        confusion_matrix(y_true, baseline_predictions, labels=labels, normalize="true"),
        confusion_matrix(y_true, bert_predictions, labels=labels, normalize="true"),
    ]
    short_names = [
        "Доступ",
        "ИИ",
        "Связь",
        "КБ",
        "Кодекс",
        "Перс. данные",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    for axis, matrix, title in zip(
        axes,
        matrices,
        ("TF-IDF + Logistic Regression", "Frozen RuBERT + Logistic Regression"),
    ):
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        axis.set_title(title, fontsize=14, weight="bold")
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
                    fontsize=9,
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Классификация статей по исходному нормативному акту", fontsize=16, weight="bold")
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

def run_week2_analysis(input_path: Path, output_dir: Path) -> dict:
    """Выполнить сравнение TF-IDF и frozen BERT и сохранить артефакты."""
    output_dir.mkdir(parents=True, exist_ok=True)
    articles = load_articles(input_path)
    train_indices, test_indices = split_dataset(articles)
    texts = [" ".join(article["text"].split()) for article in articles]
    labels = [article["document_id"] for article in articles]
    train_texts = [texts[index] for index in train_indices]
    test_texts = [texts[index] for index in test_indices]
    y_train = [labels[index] for index in train_indices]
    y_test = [labels[index] for index in test_indices]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[а-яёa-z][а-яёa-z-]+\b",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        sublinear_tf=True,
        max_features=20_000,
    )
    train_tfidf = vectorizer.fit_transform(train_texts)
    test_tfidf = vectorizer.transform(test_texts)
    baseline_classifier = build_classifier()
    baseline_classifier.fit(train_tfidf, y_train)
    baseline_predictions = baseline_classifier.predict(test_tfidf).tolist()

    bert_embeddings, chunk_counts, cache_used = load_or_create_bert_embeddings(
        articles, output_dir
    )
    bert_classifier = build_classifier()
    bert_classifier.fit(bert_embeddings[train_indices], y_train)
    bert_predictions = bert_classifier.predict(bert_embeddings[test_indices]).tolist()

    metrics = {
        "TF-IDF + Logistic Regression": calculate_metrics(y_test, baseline_predictions),
        "Frozen RuBERT + Logistic Regression": calculate_metrics(y_test, bert_predictions),
    }
    comparison_rows = [
        {"model": name, **values} for name, values in metrics.items()
    ]
    _write_csv(
        output_dir / "model_comparison.csv",
        comparison_rows,
        ["model", "accuracy", "precision_macro", "recall_macro", "f1_macro"],
    )

    split_rows = []
    train_set = set(int(index) for index in train_indices)
    for index, article in enumerate(articles):
        split_rows.append(
            {
                "article_id": article["article_id"],
                "document_id": article["document_id"],
                "split": "train" if index in train_set else "test",
                "clean_word_count": article["clean_word_count"],
                "bert_chunks": chunk_counts[index],
            }
        )
    _write_csv(
        output_dir / "dataset_split.csv",
        split_rows,
        ["article_id", "document_id", "split", "clean_word_count", "bert_chunks"],
    )

    prediction_rows = []
    for position, article_index in enumerate(test_indices):
        article = articles[int(article_index)]
        prediction_rows.append(
            {
                "article_id": article["article_id"],
                "document_title": article["document_title"],
                "true_label": y_test[position],
                "tfidf_prediction": baseline_predictions[position],
                "tfidf_correct": baseline_predictions[position] == y_test[position],
                "bert_prediction": bert_predictions[position],
                "bert_correct": bert_predictions[position] == y_test[position],
                "preview": texts[int(article_index)][:260],
            }
        )
    _write_csv(
        output_dir / "test_predictions.csv",
        prediction_rows,
        list(prediction_rows[0]),
    )

    class_labels = sorted(set(labels))
    reports = {}
    per_class_rows = []
    for model_name, predictions in (
        ("TF-IDF + Logistic Regression", baseline_predictions),
        ("Frozen RuBERT + Logistic Regression", bert_predictions),
    ):
        report = classification_report(
            y_test,
            predictions,
            labels=class_labels,
            output_dict=True,
            zero_division=0,
        )
        reports[model_name] = report
        for label in class_labels:
            per_class_rows.append(
                {
                    "model": model_name,
                    "label": label,
                    "label_name": LABEL_NAMES[label],
                    "precision": round(float(report[label]["precision"]), 4),
                    "recall": round(float(report[label]["recall"]), 4),
                    "f1": round(float(report[label]["f1-score"]), 4),
                    "support": int(report[label]["support"]),
                }
            )
    (output_dir / "classification_reports.json").write_text(
        json.dumps(reports, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        output_dir / "per_class_metrics.csv",
        per_class_rows,
        ["model", "label", "label_name", "precision", "recall", "f1", "support"],
    )
    plot_confusion_matrices(
        y_test,
        baseline_predictions,
        bert_predictions,
        class_labels,
        output_dir / "confusion_matrices.png",
    )

    joblib.dump(vectorizer, output_dir / "tfidf_vectorizer.joblib")
    joblib.dump(baseline_classifier, output_dir / "tfidf_logreg.joblib")
    joblib.dump(bert_classifier, output_dir / "bert_feature_logreg.joblib")

    summary = {
        "week": 2,
        "task": "Классификация статьи по исходному нормативному акту",
        "article_count": len(articles),
        "train_count": len(train_indices),
        "test_count": len(test_indices),
        "class_count": len(class_labels),
        "minimum_words": MIN_WORDS,
        "test_size": TEST_SIZE,
        "seed": SEED,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "bert_frozen": True,
        "bert_embedding_dimension": int(bert_embeddings.shape[1]),
        "tfidf_vocabulary_size": len(vectorizer.vocabulary_),
        "embedding_cache_used": cache_used,
        "metrics": metrics,
        "limitations": [
            "Метки получены из источника статьи.",
            "Разбиение выполнено по статьям;",
            "BERT используется только как frozen.",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary

