from __future__ import annotations

import csv
import gc
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .rk2_week2 import LABEL_NAMES, MODEL_NAME, MODEL_REVISION, SEED, load_articles, split_dataset
from .rk2_week3 import (
    BATCH_SIZE,
    MAX_TOKENS,
    WARMUP_RATIO,
    WEIGHT_DECAY,
    evaluate_fine_tuned_model,
    set_reproducible_seed,
    split_train_validation,
    train_model,
)


EXPERIMENT_EPOCHS = 2
LEARNING_RATES = (1e-5, 3e-5, 5e-5)
DATA_FRACTIONS = (0.50, 0.75, 1.00)
BASE_LEARNING_RATE = 3e-5
ERROR_SAMPLE_SIZE = 10


def stratified_subset_indices(
    articles: list[dict],
    indices: np.ndarray,
    fraction: float,
    *,
    seed: int = SEED,
) -> np.ndarray:
    if not 0 < fraction <= 1:
        raise ValueError("Доля обучающих данных должна находиться в интервале (0, 1]")
    if fraction == 1:
        return np.sort(np.asarray(indices, dtype=int))

    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for index in indices:
        label = articles[int(index)]["document_id"]
        groups.setdefault(label, []).append(int(index))

    selected: list[int] = []
    for label in sorted(groups):
        candidates = np.asarray(groups[label], dtype=int)
        rng.shuffle(candidates)
        count = max(1, int(round(len(candidates) * fraction)))
        selected.extend(candidates[:count].tolist())
    return np.asarray(sorted(selected), dtype=int)


def build_experiment_specs() -> list[dict]:
    specs = [
        {
            "experiment_id": f"lr_{rate:.0e}_data_100".replace("-", "m"),
            "learning_rate": rate,
            "data_fraction": 1.0,
        }
        for rate in LEARNING_RATES
    ]
    specs.extend(
        {
            "experiment_id": f"lr_3em05_data_{int(fraction * 100)}",
            "learning_rate": BASE_LEARNING_RATE,
            "data_fraction": fraction,
        }
        for fraction in DATA_FRACTIONS
        if fraction < 1.0
    )
    return specs


def classify_error_cause(
    article: dict,
    training_counts: Counter,
) -> tuple[str, str]:
    text = " ".join(article["text"].lower().split())
    word_count = int(article.get("clean_word_count", len(text.split())))
    median_count = float(np.median(list(training_counts.values())))

    if "�" in text or "\x00" in text:
        return "шум в тексте", "обнаружены необычные или повреждённые символы"
    if word_count < 80:
        return "короткий общий контекст", f"в статье только {word_count} слов"
    reference_markers = ("в соответствии с", "согласно", "предусмотренн", "настоящего кодекса")
    if sum(marker in text for marker in reference_markers) >= 2:
        return "междокументные ссылки", "текст активно ссылается на другие нормы и кодекс"
    if training_counts[article["document_id"]] < median_count:
        return "дисбаланс классов", "истинный класс представлен в train слабее медианного класса"
    return "неоднозначность контекста", "лексика статьи подходит нескольким тематическим актам"


def build_error_analysis(
    articles: list[dict],
    fit_indices: np.ndarray,
    week3_predictions_path: Path,
) -> tuple[list[dict], dict[str, int]]:
    """Выбрать десять уверенных ошибок базовой модели третьей недели."""
    with week3_predictions_path.open(encoding="utf-8-sig", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    errors = [row for row in predictions if row["correct"].strip().lower() != "true"]
    errors.sort(key=lambda row: float(row["confidence"]), reverse=True)
    if len(errors) < ERROR_SAMPLE_SIZE:
        raise ValueError(
            f"У базовой модели найдено только {len(errors)} ошибок; нужно минимум {ERROR_SAMPLE_SIZE}"
        )

    by_id = {article["article_id"]: article for article in articles}
    training_counts = Counter(articles[int(index)]["document_id"] for index in fit_indices)
    rows = []
    causes: Counter[str] = Counter()
    for rank, prediction in enumerate(errors[:ERROR_SAMPLE_SIZE], 1):
        article = by_id[prediction["article_id"]]
        cause, explanation = classify_error_cause(article, training_counts)
        causes[cause] += 1
        rows.append(
            {
                "rank": rank,
                "article_id": article["article_id"],
                "article_title": article.get("article_title", ""),
                "true_label": article["document_id"],
                "true_label_ru": LABEL_NAMES.get(article["document_id"], article["document_id"]),
                "predicted_label": prediction["predicted_label"],
                "predicted_label_ru": LABEL_NAMES.get(
                    prediction["predicted_label"], prediction["predicted_label"]
                ),
                "confidence": prediction["confidence"],
                "hypothesized_cause": cause,
                "cause_explanation": explanation,
                "word_count": article.get("clean_word_count", len(article["text"].split())),
                "text_preview": " ".join(article["text"].split())[:420],
            }
        )
    return rows, dict(sorted(causes.items()))


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"Нечего записывать в {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_experiment_metrics(rows: list[dict], output_path: Path) -> None:
    controlled = [row for row in rows if row["source"] == "week4_controlled"]
    labels = [
        f"LR {row['learning_rate']:.0e}\nДанные {row['data_fraction']:.0%}"
        for row in controlled
    ]
    x = np.arange(len(controlled))
    width = 0.19
    metrics = [
        ("accuracy", "Accuracy"),
        ("precision_macro", "Precision"),
        ("recall_macro", "Recall"),
        ("f1_macro", "F1"),
    ]
    fig, axis = plt.subplots(figsize=(13, 6.5), constrained_layout=True)
    for offset, (key, title) in enumerate(metrics):
        values = [row[key] for row in controlled]
        axis.bar(x + (offset - 1.5) * width, values, width, label=title)
    axis.set_xticks(x, labels, fontsize=8)
    axis.set_ylim(0, 1.02)
    axis.set_ylabel("Значение метрики")
    axis.set_title("Сводное сравнение экспериментов", fontsize=15, weight="bold")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncols=4, loc="lower center")
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_factor_dynamics(rows: list[dict], output_dir: Path) -> None:
    controlled = [row for row in rows if row["source"] == "week4_controlled"]
    lr_rows = sorted(
        (row for row in controlled if row["data_fraction"] == 1.0),
        key=lambda row: row["learning_rate"],
    )
    data_rows = sorted(
        (row for row in controlled if row["learning_rate"] == BASE_LEARNING_RATE),
        key=lambda row: row["data_fraction"],
    )

    fig, axis = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    axis.plot(
        [row["learning_rate"] for row in lr_rows],
        [row["f1_macro"] for row in lr_rows],
        marker="o",
        linewidth=2,
        label="Macro-F1",
    )
    axis.plot(
        [row["learning_rate"] for row in lr_rows],
        [row["accuracy"] for row in lr_rows],
        marker="s",
        linewidth=2,
        label="Accuracy",
    )
    axis.set_xscale("log")
    axis.set_xticks(list(LEARNING_RATES), [f"{value:.0e}" for value in LEARNING_RATES])
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Learning rate")
    axis.set_ylabel("Метрика на test")
    axis.set_title("Влияние learning rate при 100% train", weight="bold")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output_dir / "learning_rate_dynamics.png", dpi=170, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 5.2), constrained_layout=True)
    percentages = [int(row["data_fraction"] * 100) for row in data_rows]
    axis.plot(percentages, [row["f1_macro"] for row in data_rows], marker="o", label="Macro-F1")
    axis.plot(percentages, [row["accuracy"] for row in data_rows], marker="s", label="Accuracy")
    axis.set_xticks(percentages, [f"{value}%" for value in percentages])
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Доля доступных обучающих статей")
    axis.set_ylabel("Метрика на test")
    axis.set_title("Влияние объёма обучающей выборки", weight="bold")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.savefig(output_dir / "data_size_dynamics.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_error_causes(causes: dict[str, int], output_path: Path) -> None:
    labels = list(causes)
    values = [causes[label] for label in labels]
    fig, axis = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    bars = axis.barh(labels, values, color="#0f766e")
    axis.bar_label(bars, padding=4)
    axis.set_xlim(0, max(values) + 1)
    axis.set_xlabel("Число ошибок в выборке из 10 примеров")
    axis.set_title("Гипотезы о причинах ошибок", fontsize=14, weight="bold")
    axis.grid(axis="x", alpha=0.25)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

def run_week4_analysis(
    input_path: Path,
    week3_summary_path: Path,
    week3_predictions_path: Path,
    output_dir: Path,
) -> dict:
    """Провести пять контролируемых опытов и сформировать анализ ошибок."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    set_reproducible_seed()
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / "training_logs"
    logs_dir.mkdir(exist_ok=True)

    articles = load_articles(input_path)
    train_pool_indices, test_indices = split_dataset(articles)
    fit_indices, validation_indices = split_train_validation(articles, train_pool_indices)
    labels = sorted({article["document_id"] for article in articles})
    label_to_id = {label: index for index, label in enumerate(labels)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=MODEL_REVISION)
    week3_summary = json.loads(week3_summary_path.read_text(encoding="utf-8"))

    baseline_metrics = week3_summary["comparison"]["after_fine_tuning"]
    result_rows = [
        {
            "experiment_id": "week3_baseline",
            "source": "week3_baseline",
            "data_fraction": 1.0,
            "train_articles": len(fit_indices),
            "learning_rate": week3_summary["hyperparameters"]["learning_rate"],
            "epochs": week3_summary["hyperparameters"]["epochs"],
            "best_epoch": week3_summary["best_epoch"],
            "validation_f1_macro": week3_summary["best_validation_f1_macro"],
            "runtime_seconds": "",
            **baseline_metrics,
        }
    ]
    combined_history = []
    selected = None
    selected_validation_f1 = -1.0

    for spec in build_experiment_specs():
        set_reproducible_seed()
        subset_indices = stratified_subset_indices(
            articles, fit_indices, spec["data_fraction"], seed=SEED
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            num_labels=len(labels),
            id2label=id_to_label,
            label2id=label_to_id,
        )
        started = time.perf_counter()
        history, best = train_model(
            model,
            tokenizer,
            articles,
            subset_indices,
            validation_indices,
            label_to_id,
            id_to_label,
            epochs=EXPERIMENT_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=spec["learning_rate"],
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
            max_tokens=MAX_TOKENS,
        )
        test_result = evaluate_fine_tuned_model(
            model,
            tokenizer,
            articles,
            test_indices,
            label_to_id,
            id_to_label,
            best["loss_function"],
            batch_size=BATCH_SIZE,
            max_tokens=MAX_TOKENS,
        )
        runtime = round(time.perf_counter() - started, 2)
        row = {
            "experiment_id": spec["experiment_id"],
            "source": "week4_controlled",
            "data_fraction": spec["data_fraction"],
            "train_articles": len(subset_indices),
            "learning_rate": spec["learning_rate"],
            "epochs": EXPERIMENT_EPOCHS,
            "best_epoch": best["epoch"],
            "validation_f1_macro": round(float(best["validation_f1_macro"]), 4),
            "runtime_seconds": runtime,
            **test_result["metrics"],
        }
        result_rows.append(row)
        for history_row in history:
            combined_history.append({"experiment_id": spec["experiment_id"], **history_row})
        _write_csv(logs_dir / f"{spec['experiment_id']}.csv", history)

        if float(best["validation_f1_macro"]) > selected_validation_f1:
            selected_validation_f1 = float(best["validation_f1_macro"])
            selected = dict(row)
            selected_dir = output_dir / "selected_model"
            model.save_pretrained(selected_dir, safe_serialization=True)
            tokenizer.save_pretrained(selected_dir)

        del model, best, test_result
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if selected is None:
        raise RuntimeError("Не удалось выбрать экспериментальную модель")

    error_rows, error_causes = build_error_analysis(
        articles, fit_indices, week3_predictions_path
    )
    _write_csv(output_dir / "experiment_results.csv", result_rows)
    _write_csv(output_dir / "training_history.csv", combined_history)
    _write_csv(output_dir / "error_analysis_10.csv", error_rows)
    plot_experiment_metrics(result_rows, output_dir / "experiment_metrics.png")
    plot_factor_dynamics(result_rows, output_dir)
    plot_error_causes(error_causes, output_dir / "error_causes.png")

    summary = {
        "week": 4,
        "task": "Экспериментальное исследование и анализ ошибок fine-tuned RuBERT",
        "article_count": len(articles),
        "fit_count": len(fit_indices),
        "validation_count": len(validation_indices),
        "test_count": len(test_indices),
        "controlled_run_count": len(build_experiment_specs()),
        "controlled_epochs": EXPERIMENT_EPOCHS,
        "learning_rates": list(LEARNING_RATES),
        "data_fractions": list(DATA_FRACTIONS),
        "week3_baseline": baseline_metrics,
        "selected_by_validation": selected,
        "error_sample_size": len(error_rows),
        "error_causes": error_causes,
        "test_isolation": (
            "Одинаковый test использован только для итоговой оценки; конфигурация выбрана по validation."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
