"""Собрать выполненный Notebook четвёртой недели с таблицами и графиками."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "rk2_week4"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "rk2_week4_experiments_errors.ipynb"


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
            """# Второй РК — неделя 4: эксперименты и анализ ошибок

Исследуется влияние learning rate и объёма train на fine-tuned RuBERT. Затем
разбираются десять реальных ошибок базовой модели третьей недели.
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

from lawankz.rk2_week4 import (
    build_experiment_specs,
    run_week4_analysis,
    stratified_subset_indices,
)

RESULTS_DIR = PROJECT_ROOT / "results" / "rk2_week4"
summary = json.loads((RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
summary"""
        ),
        nbf.v4.new_markdown_cell("## 1. План контролируемых экспериментов"),
        nbf.v4.new_code_cell("pd.DataFrame(build_experiment_specs())"),
        nbf.v4.new_markdown_cell(
            """Опыты используют одинаковые validation/test, seed, batch size и
число эпох. Центральная конфигурация `3e-5 + 100%` общая для двух серий.
Стратифицированный отбор сохраняет каждый класс при уменьшении train.
"""
        ),
        nbf.v4.new_code_cell("print(inspect.getsource(stratified_subset_indices))"),
        nbf.v4.new_markdown_cell(
            """## 2. Код полного запуска

Повторное обучение выключено, чтобы открытие Notebook не занимало несколько
минут. Код запуска оставлен исполняемым и воспроизводимым.
"""
        ),
        nbf.v4.new_code_cell(
            """RUN_TRAINING = False
if RUN_TRAINING:
    summary = run_week4_analysis(
        PROJECT_ROOT / "results" / "weeks_1_4" / "processed_articles.jsonl",
        PROJECT_ROOT / "results" / "rk2_week3" / "summary.json",
        PROJECT_ROOT / "results" / "rk2_week3" / "test_predictions.csv",
        RESULTS_DIR,
    )
else:
    print("Загружены сохранённые результаты пяти выполненных запусков.")"""
        ),
        nbf.v4.new_markdown_cell("## 3. Сводная таблица Accuracy, Precision, Recall и F1"),
        nbf.v4.new_code_cell(
            """results = pd.read_csv(RESULTS_DIR / "experiment_results.csv")
results"""
        ),
        nbf.v4.new_code_cell(
            """display(Image(filename=str(RESULTS_DIR / "experiment_metrics.png"), width=1050))"""
        ),
        nbf.v4.new_markdown_cell("## 4. Динамика метрик"),
        nbf.v4.new_code_cell(
            """display(Image(filename=str(RESULTS_DIR / "learning_rate_dynamics.png"), width=700))
display(Image(filename=str(RESULTS_DIR / "data_size_dynamics.png"), width=700))"""
        ),
        nbf.v4.new_markdown_cell("## 5. Десять ошибок базовой модели"),
        nbf.v4.new_code_cell(
            """errors = pd.read_csv(RESULTS_DIR / "error_analysis_10.csv")
errors[[
    "article_id", "true_label_ru", "predicted_label_ru", "confidence",
    "hypothesized_cause", "cause_explanation", "text_preview",
]]"""
        ),
        nbf.v4.new_code_cell(
            """display(Image(filename=str(RESULTS_DIR / "error_causes.png"), width=800))"""
        ),
        nbf.v4.new_markdown_cell(report),
        nbf.v4.new_markdown_cell(
            f"""## Зафиксированный итог

- контролируемых запусков: {summary['controlled_run_count']};
- размеры fit/validation/test: {summary['fit_count']}/{summary['validation_count']}/{summary['test_count']};
- разобрано ошибок: {summary['error_sample_size']};
- модель выбрана по validation, test не использован для выбора.
"""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
