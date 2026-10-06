from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .markdown import render_markdown
from .preprocessing import analyze_text


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Выделить статьи и кандидаты в нормативные требования из TXT-файла."
    )
    parser.add_argument("input", type=Path, help="path to russian TXT-file")
    parser.add_argument("--output", type=Path, help="Куда сохранить результат (.json или .md)")
    parser.add_argument("--format", choices=("json", "md"), help="Формат вывода; по умолчанию определяется расширением --output")
    args = parser.parse_args()

    try:
        text = args.input.read_text(encoding="utf-8")
        result = analyze_text(text)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Ошибка: {exc}\n")

    result["source_file"] = args.input.name
    output_format = args.format or ("md" if args.output and args.output.suffix.lower() == ".md" else "json")
    serialized = (
        render_markdown(result)
        if output_format == "md"
        else json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    if args.output:
        try:
            args.output.write_text(serialized, encoding="utf-8")
        except OSError as exc:
            parser.exit(1, f"Ошибка сохранения: {exc}\n")
        print(f"Результат сохранён: {args.output}", file=sys.stderr)
    else:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
