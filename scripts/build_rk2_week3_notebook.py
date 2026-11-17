"""Собрать выполненный Notebook третьей недели без повторного долгого обучения."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "rk2_week3"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "rk2_week3_fine_tuning.ipynb"


def main() -> int:
    summary = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
    report = (RESULTS_DIR / "report.md").read_text(encoding="utf-8")
    notebook = nbf.v4.new_notebook()
    notebook["metadata"].update(
        {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.13"},
        }
    )
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            """# Второй РК — неделя 3: fine-tuning RuBERT

Цель — адаптировать предобученную модель к классификации статей по шести
исходным нормативным актам и честно сравнить её с frozen BERT второй недели.
"""
        ),
        nbf.v4.new_code_cell(
            """from pathlib import Path
import inspect
import json
import sys

import pandas as pd
from IPython.display import Image, display

PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lawankz.rk2_week3 import run_week3_analysis, train_model

RESULTS_DIR = PROJECT_ROOT / "results" / "rk2_week3"
summary = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
summary"""
        ),
        nbf.v4.new_markdown_cell(
            """## 1. Код обучения

Ниже выводится реальная функция проекта. Внутри есть `AdamW`, взвешенная
cross-entropy, обратное распространение ошибки, clipping градиента и scheduler.
"""
        ),
        nbf.v4.new_code_cell("print(inspect.getsource(train_model))"),
        nbf.v4.new_markdown_cell(
            """## 2. Воспроизводимый запуск

Флаг оставлен `False`, потому что модель уже обучена, а повтор занимает несколько
минут на CPU. Для полного повторения достаточно переключить его в `True`.
"""
        ),
        nbf.v4.new_code_cell(
            """RUN_TRAINING = False
if RUN_TRAINING:
    summary = run_week3_analysis(
        PROJECT_ROOT / "results" / "weeks_1_4" / "processed_articles.jsonl",
        PROJECT_ROOT / "results" / "rk2_week2" / "summary.json",
        RESULTS_DIR,
    )
else:
    print("Используются сохранённые результаты честно выполненного запуска.")"""
        ),
        nbf.v4.new_markdown_cell("## 3. Гиперпараметры и ход обучения"),
        nbf.v4.new_code_cell(
            """config = json.loads((RESULTS_DIR / "training_config.json").read_text(encoding="utf-8"))
pd.Series(config, name="value").to_frame()"""
        ),
        nbf.v4.new_code_cell(
            """history = pd.read_csv(RESULTS_DIR / "training_log.csv")
history"""
        ),
        nbf.v4.new_code_cell(
            """display(Image(filename=str(RESULTS_DIR / "learning_curves.png"), width=950))"""
        ),
        nbf.v4.new_markdown_cell("## 4. Результат «до / после»"),
        nbf.v4.new_code_cell(
            """comparison = pd.read_csv(RESULTS_DIR / "before_after_comparison.csv")
comparison"""
        ),
        nbf.v4.new_code_cell(
            """display(Image(filename=str(RESULTS_DIR / "confusion_matrix.png"), width=760))"""
        ),
        nbf.v4.new_markdown_cell(report),
        nbf.v4.new_markdown_cell(
            f"""## Зафиксированный итог

- train pool: {summary['train_pool_count']};
- fit/validation/test: {summary['fit_count']}/{summary['validation_count']}/{summary['test_count']};
- лучшая эпоха: {summary['best_epoch']};
- validation macro-F1: {summary['best_validation_f1_macro']:.4f};
- test macro-F1 после fine-tuning: {summary['comparison']['after_fine_tuning']['f1_macro']:.4f}.
"""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
