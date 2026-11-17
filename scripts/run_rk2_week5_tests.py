"""Подготовить пять контрольных текстов и проверить модель пятой недели."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lawankz.rk2_week2 import load_articles  # noqa: E402
from lawankz.rk2_week5 import (  # noqa: E402
    load_model_bundle,
    predict_text,
    select_control_examples,
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    output_dir = PROJECT_ROOT / "results" / "rk2_week5"
    output_dir.mkdir(parents=True, exist_ok=True)
    articles = load_articles(
        PROJECT_ROOT / "results" / "weeks_1_4" / "processed_articles.jsonl"
    )
    examples = select_control_examples(articles, count=5)
    (output_dir / "control_examples.json").write_text(
        json.dumps(examples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    tokenizer, model = load_model_bundle(
        PROJECT_ROOT / "results" / "rk2_week4" / "selected_model"
    )
    rows = []
    for example in examples:
        result = predict_text(example["text"], tokenizer, model)
        rows.append(
            {
                "sample_id": example["sample_id"],
                "article_id": example["article_id"],
                "expected_label": example["expected_label"],
                "expected_label_ru": example["expected_label_ru"],
                "predicted_label": result["predicted_label"],
                "predicted_label_ru": result["predicted_label_ru"],
                "correct": result["predicted_label"] == example["expected_label"],
                "confidence": result["confidence"],
                "confidence_level": result["confidence_level"],
                "word_count": result["word_count"],
                "chunk_count": result["chunk_count"],
                "screenshot": f"screenshots/test_{example['sample_id']}.png",
            }
        )
    with (output_dir / "control_test_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    correct_count = sum(row["correct"] for row in rows)
    summary = {
        "week": 5,
        "framework": "Streamlit",
        "model_path": "results/rk2_week4/selected_model",
        "control_example_count": len(rows),
        "correct_count": correct_count,
        "control_accuracy": round(correct_count / len(rows), 4),
        "screenshots_expected": len(rows),
        "features": [
            "текстовое поле",
            "кнопка анализа",
            "предсказанный класс",
            "уверенность",
            "диаграмма вероятностей",
            "пять контрольных примеров",
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
