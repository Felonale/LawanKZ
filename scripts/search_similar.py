"""Командный прототип поиска статей по смысловому запросу."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from gensim.models import Word2Vec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lawankz.coursework import preprocess_text  # noqa: E402
from lawankz.embeddings import load_processed_articles, search_by_vector  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Найти похожие статьи НПА")
    parser.add_argument("query", help="Запрос на русском языке")
    parser.add_argument("--top", type=int, default=5, help="Число результатов")
    args = parser.parse_args()
    if args.top < 1:
        parser.error("--top должен быть положительным")

    output_dir = PROJECT_ROOT / "results" / "weeks_5_6"
    model = Word2Vec.load(str(output_dir / "word2vec.model")).wv
    idf = json.loads((output_dir / "idf.json").read_text(encoding="utf-8"))
    articles = load_processed_articles(output_dir / "search_articles.jsonl")
    article_vectors = np.load(output_dir / "article_vectors.npy")
    lemmas = preprocess_text(args.query)["lemmas"]
    results = search_by_vector(lemmas, model, idf, articles, article_vectors, top_k=args.top)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(f"Запрос: {args.query}\nЛеммы: {', '.join(lemmas)}\n")
    for rank, result in enumerate(results, start=1):
        heading = result["article_title"] or "Без заголовка"
        print(
            f"{rank}. {result['document_title']} — статья {result['article_number']}: {heading}\n"
            f"   сходство: {result['similarity']:.4f}\n   {result['preview']}\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
