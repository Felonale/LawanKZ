"""Собрать Jupyter Notebook с выполненным экспериментом второй недели."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "rk2_week2"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "rk2_week2_pretrained_bert.ipynb"


def main() -> int:
    summary = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
    report = (RESULTS_DIR / "report.md").read_text(encoding="utf-8")
    notebook = nbf.v4.new_notebook()
    notebook["metadata"].update(
        {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.13"},
        }
    )
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Второй РК — неделя 2: предобученный BERT

**Задача:** по тексту статьи определить, из какого тематического нормативного
акта она получена.

Сравниваются две системы:

1. TF-IDF + Logistic Regression — классическая базовая модель;
2. frozen RuBERT + Logistic Regression — предобученный BERT используется только
   как поставщик признаков, его веса не изменяются.

Используется модель
[`cointegrated/rubert-tiny2`](https://huggingface.co/cointegrated/rubert-tiny2).
"""
        ),
        nbf.v4.new_markdown_cell(
            """## 1. Почему это feature extraction, а не fine-tuning

Текст разбивается на фрагменты, токенизируется и передаётся в `AutoModel` внутри
`torch.no_grad()`. Из последнего скрытого слоя берётся CLS-вектор. Градиенты не
вычисляются, оптимизатор отсутствует, веса BERT не меняются. Обучается только
небольшая Logistic Regression поверх готовых 312-мерных признаков.
"""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import json
import sys

import pandas as pd
from IPython.display import Image, display

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lawankz.rk2_week2 import run_week2_analysis

RESULTS_DIR = PROJECT_ROOT / "results" / "rk2_week2"
summary = run_week2_analysis(
    PROJECT_ROOT / "results" / "weeks_1_4" / "processed_articles.jsonl",
    RESULTS_DIR,
)
summary"""
        ),
        nbf.v4.new_markdown_cell("## 2. Разбиение данных"),
        nbf.v4.new_code_cell(
            """split = pd.read_csv(RESULTS_DIR / "dataset_split.csv")
pd.crosstab(split["document_id"], split["split"], margins=True)"""
        ),
        nbf.v4.new_markdown_cell(
            """Разбиение стратифицировано: доля каждого из шести актов сохраняется
в train и test. `random_state=42` фиксирует результат. TF-IDF-словарь и оба
классификатора обучаются только на train; тестовые метки используются лишь для
финального расчёта метрик.
"""
        ),
        nbf.v4.new_markdown_cell("## 3. Сравнение моделей"),
        nbf.v4.new_code_cell(
            """comparison = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
comparison"""
        ),
        nbf.v4.new_code_cell(
            """display(Image(filename=str(RESULTS_DIR / "confusion_matrices.png"), width=1100))"""
        ),
        nbf.v4.new_markdown_cell(
            """В матрице ошибок каждая строка — настоящий класс, каждый столбец —
предсказание. Число на диагонали показывает долю правильных ответов класса.
Например, `1.00` означает, что все тестовые статьи данного акта распознаны
правильно.
"""
        ),
        nbf.v4.new_markdown_cell("## 4. Метрики отдельных классов"),
        nbf.v4.new_code_cell(
            """per_class = pd.read_csv(RESULTS_DIR / "per_class_metrics.csv")
per_class.pivot(index="label_name", columns="model", values="f1").round(4)"""
        ),
        nbf.v4.new_markdown_cell("## 5. Примеры ошибок"),
        nbf.v4.new_code_cell(
            """predictions = pd.read_csv(RESULTS_DIR / "test_predictions.csv")
errors = predictions.loc[
    ~predictions["bert_correct"],
    ["article_id", "true_label", "bert_prediction", "preview"],
]
errors.head(10)"""
        ),
        nbf.v4.new_markdown_cell(report),
        nbf.v4.new_markdown_cell(
            f"""## Техническая фиксация

- статей после фильтрации: {summary['article_count']};
- train/test: {summary['train_count']}/{summary['test_count']};
- классов: {summary['class_count']};
- BERT: `{summary['model_name']}`;
- ревизия: `{summary['model_revision']}`;
- размер BERT-вектора: {summary['bert_embedding_dimension']};
- размер словаря TF-IDF: {summary['tfidf_vocabulary_size']};
- веса BERT заморожены: `{summary['bert_frozen']}`.
"""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
