"""Собрать Jupyter Notebook с выполненным экспериментом первой недели."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "rk2_week1"
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "rk2_week1_attention_transformer.ipynb"


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
            """# Второй РК — неделя 1: Transformer и Attention

**Проект:** анализ текстов законов и нормативных правовых актов Республики Казахстан.

В ноутбуке выполнены все пункты первой недели: выбраны 15 текстов, получены
Transformer-эмбеддинги, проведено сравнение с TF-IDF, разобрано контекстное
значение слова «оператор» и сформулированы выводы.

Используется многоязычная модель
[`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2).
"""
        ),
        nbf.v4.new_markdown_cell(
            r"""## 1. Коротко о механизме Attention

Пусть для каждого токена построены векторы запроса $Q$, ключа $K$ и значения $V$.
Веса внимания вычисляются как

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V.$$

Скалярные произведения определяют, какие элементы контекста важнее для текущего
токена. В Transformer несколько голов внимания работают параллельно, а позиционные
признаки сохраняют информацию о порядке. В отличие от TF-IDF, координаты итогового
вектора не соответствуют отдельным словам и зависят от всего контекста.
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

from lawankz.rk2_week1 import run_week1_analysis

RESULTS_DIR = PROJECT_ROOT / "results" / "rk2_week1"
summary = run_week1_analysis(
    PROJECT_ROOT / "results" / "weeks_1_4" / "processed_articles.jsonl",
    RESULTS_DIR,
)
summary["metrics"]"""
        ),
        nbf.v4.new_markdown_cell("## 2. Выбранные тексты"),
        nbf.v4.new_code_cell(
            """selected = pd.read_csv(RESULTS_DIR / "selected_texts.csv")
selected[[
    "label", "document_title", "article_number",
    "clean_word_count", "transformer_chunks"
]]"""
        ),
        nbf.v4.new_markdown_cell(
            """Выборка сбалансирована: по три статьи из пяти актов. Очень короткие
определения и аномально длинные статьи исключены. Длинные тексты разбиваются по
границам предложений примерно на 90 слов; вектор статьи — нормированное среднее
векторов фрагментов. Это предотвращает молчаливое отсечение большей части статьи
ограничением длины Transformer.
"""
        ),
        nbf.v4.new_markdown_cell("## 3. TF-IDF и Transformer на одинаковых данных"),
        nbf.v4.new_code_cell(
            """metrics = pd.DataFrame(summary["metrics"]).T
metrics"""
        ),
        nbf.v4.new_code_cell(
            """display(Image(filename=str(RESULTS_DIR / "tfidf_vs_transformer.png"), width=1000))"""
        ),
        nbf.v4.new_markdown_cell(
            """`Precision@2` — доля статей того же исходного акта среди двух
ближайших соседей. `Hit rate@2` показывает, встретился ли там хотя бы один такой
сосед. Принадлежность к одному акту — лишь автоматическая приближённая разметка,
а не юридическая оценка релевантности.
"""
        ),
        nbf.v4.new_markdown_cell("## 4. Контекстное значение слова «оператор»"),
        nbf.v4.new_code_cell(
            """context = json.loads((RESULTS_DIR / "context_analysis.json").read_text(encoding="utf-8"))
pd.DataFrame(context["sentences"])[["context", "sentence"]]"""
        ),
        nbf.v4.new_code_cell(
            """pd.DataFrame(
    {
        "пара контекстов": [
            "персональные данные — телекоммуникации",
            "персональные данные — кинопроизводство",
            "телекоммуникации — кинопроизводство",
        ],
        "косинус скрытых векторов слова": [
            context["legal_vs_telecom_cosine"],
            context["legal_vs_cinema_cosine"],
            context["telecom_vs_cinema_cosine"],
        ],
    }
)"""
        ),
        nbf.v4.new_markdown_cell(
            """Одинаковое написание «оператор» получает разные скрытые векторы,
потому что self-attention учитывает соседние токены. Однако три предложения —
демонстрация механизма, а не статистически надёжный тест качества различения
значений. Неожиданный порядок косинусов нужно честно фиксировать, а не подгонять
под ожидаемый вывод.
"""
        ),
        nbf.v4.new_markdown_cell(report),
        nbf.v4.new_markdown_cell(
            f"""## Техническая фиксация запуска

- модель: `{summary['model_name']}`;
- ревизия модели: `{summary['model_revision']}`;
- размер Transformer-вектора: {summary['transformer_dimension']};
- размер TF-IDF-вектора: {summary['tfidf_dimension']};
- число текстов: {summary['selected_text_count']}.
"""
        ),
    ]
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, NOTEBOOK_PATH)
    print(NOTEBOOK_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
