"""Пересчитать учебный анализ по сохранённому корпусу без обращения к сайту."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lawankz.corpus import load_documents  # noqa: E402
from lawankz.coursework import run_analysis  # noqa: E402


def main() -> int:
    documents = load_documents(PROJECT_ROOT / "data" / "processed" / "documents.jsonl")
    summary = run_analysis(documents, PROJECT_ROOT / "results" / "weeks_1_4")
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(
        f"Готово: {summary['document_count']} актов, {summary['article_count']} статей, "
        f"{summary['raw_word_count']} слов; матрица BoW {summary['bow_shape']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
