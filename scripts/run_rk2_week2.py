"""Запустить эксперимент второй недели второго рубежного контроля."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from lawankz.rk2_week2 import run_week2_analysis  # noqa: E402


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    summary = run_week2_analysis(
        PROJECT_ROOT / "results" / "weeks_1_4" / "processed_articles.jsonl",
        PROJECT_ROOT / "results" / "rk2_week2",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

