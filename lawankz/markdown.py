"""Превращение JSON-результата анализа в простой Markdown-отчёт."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def render_markdown(result: dict) -> str:
    """Сформировать читаемый отчёт из словаря, созданного analyze_text()."""
    if not isinstance(result, dict):
        raise ValueError("Ожидается JSON-объект с результатом анализа")

    articles = result.get("articles")
    candidates = result.get("candidate_requirements")
    if not isinstance(articles, list) or not isinstance(candidates, list):
        raise ValueError("В JSON должны быть списки articles и candidate_requirements")

    source = result.get("source_file") or "не указан"
    lines = [
        "# Анализ текста НПА",
        "",
        f"Источник: {source}",
        f"Статей: {len(articles)}",
        f"Предложений-кандидатов: {len(candidates)}",
    ]

    for article in articles:
        if not isinstance(article, dict):
            raise ValueError("Каждая статья должна быть JSON-объектом")
        number = article.get("number")
        title = article.get("title")
        heading = f"Статья {number}" if number else "Фрагмент без номера"
        if title:
            heading += f". {title}"
        lines.extend(["", f"## {heading}", ""])

        body = article.get("text") or ""
        lines.append(body if body else "Текст статьи отсутствует.")

        article_candidates = [
            item for item in candidates
            if isinstance(item, dict) and item.get("article_number") == number
        ]
        lines.extend(["", "### Найденные предложения", ""])
        if article_candidates:
            for item in article_candidates:
                sentence = item.get("sentence") or ""
                markers = item.get("markers") or []
                marker_text = ", ".join(str(marker) for marker in markers)
                lines.append(f"- {sentence} (маркер: {marker_text})")
        else:
            lines.append("Явных маркеров не найдено.")

    if result.get("warning"):
        lines.extend(["", f"Примечание: {result['warning']}"])

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Преобразовать JSON-анализ LawanKZ в Markdown.")
    parser.add_argument("input", type=Path, help="Путь к JSON-файлу")
    parser.add_argument("--output", type=Path, help="Куда сохранить Markdown")
    args = parser.parse_args()

    try:
        data = json.loads(args.input.read_text(encoding="utf-8"))
        report = render_markdown(data)
        if args.output:
            args.output.write_text(report, encoding="utf-8")
        else:
            if hasattr(sys.stdout, "reconfigure"):
                sys.stdout.reconfigure(encoding="utf-8")
            print(report, end="")
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Ошибка: {exc}\n")

    if args.output:
        print(f"Отчёт сохранён: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
