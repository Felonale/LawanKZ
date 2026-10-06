"""Скачать и подготовить небольшой официальный корпус для недель 1–4."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lawankz.corpus import build_documents, fetch_snapshots, load_sources, save_documents  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Обновить сохранённые HTML-снимки")
    args = parser.parse_args()

    sources = load_sources(PROJECT_ROOT / "data" / "sources.json")
    raw_dir = PROJECT_ROOT / "data" / "raw"
    output_dir = PROJECT_ROOT / "data" / "processed"
    fetch_snapshots(sources, raw_dir, refresh=args.refresh)
    documents = build_documents(sources, raw_dir)
    save_documents(documents, output_dir / "documents.jsonl")

    summary = {
        "document_count": len(documents),
        "total_words": sum(document["word_count"] for document in documents),
        "minimum_words_required": 50000,
        "documents": [
            {key: document[key] for key in ("id", "title", "topic", "url", "fetched_at_utc", "sha256", "word_count")}
            for document in documents
        ],
    }
    (output_dir / "corpus_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(f"Корпус: {summary['document_count']} документов, {summary['total_words']} русских слов")
    if summary["total_words"] < 50000:
        print("Внимание: порог 50 000 слов ещё не достигнут", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
