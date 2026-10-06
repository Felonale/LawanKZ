"""Обучить эмбеддинги и собрать прототип семантического поиска."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lawankz.embeddings import run_embeddings_analysis  # noqa: E402


def main() -> int:
    summary = run_embeddings_analysis(
        PROJECT_ROOT / "results" / "weeks_1_4",
        PROJECT_ROOT / "results" / "weeks_5_6",
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    metrics = summary["retrieval"]
    print(
        f"Готово: {summary['training_articles']} статей, "
        f"{summary['training_tokens']} токенов; "
        f"precision@5 поиска = {metrics['precision_at_5']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
