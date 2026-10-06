from __future__ import annotations

import re
from functools import lru_cache


ARTICLE_HEADING = re.compile(
    r"^\s*Статья\s+(?P<number>\d+(?:-\d+)?)\s*\.\s*(?P<title>.*?)\s*$",
    re.IGNORECASE,
)
REQUIREMENT_MARKERS = {
    "обязан": re.compile(r"\bобязан(?:а|о|ы)?\b", re.IGNORECASE),
    "должен": re.compile(r"\bдолж(?:ен|на|но|ны)\b", re.IGNORECASE),
    "вправе": re.compile(r"\bвправе\b", re.IGNORECASE),
    "не допускается": re.compile(r"\bне\s+допускается\b", re.IGNORECASE),
    "запрещается": re.compile(r"\bзапрещается\b", re.IGNORECASE),
    "подлежит": re.compile(r"\bподлеж(?:ит|ат)\b", re.IGNORECASE),
}


def normalize_text(text: str) -> str:
    """Убрать лишние пробелы."""
    if not text.strip():
        raise ValueError("Входной текст пуст")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.replace("\xa0", " ").splitlines()]
    return "\n".join(line for line in lines if line)


def split_articles(text: str) -> list[dict[str, str | None]]:
    """Выделить статьи по заголовкам вида «Статья 1. Название»."""
    normalized = normalize_text(text)
    articles: list[dict[str, str | None]] = []
    current: dict[str, str | None] | None = None
    body_lines: list[str] = []

    for line in normalized.splitlines():
        match = ARTICLE_HEADING.fullmatch(line)
        if match:
            if current is not None:
                current["text"] = "\n".join(body_lines).strip()
                articles.append(current)
            current = {
                "number": match.group("number"),
                "title": match.group("title"),
                "text": "",
            }
            body_lines = []
        elif current is not None:
            body_lines.append(line)

    if current is not None:
        current["text"] = "\n".join(body_lines).strip()
        articles.append(current)
    else:
        articles.append({"number": None, "title": None, "text": normalized})
    return articles


@lru_cache(maxsize=1)
def _language_model():
    try:
        import spacy

        return spacy.load("ru_core_news_sm")
    except (ImportError, OSError) as exc:
        raise OSError(
            "needed spaCy and ru_core_news_sm modules"
        ) from exc


def analyze_text(text: str) -> dict:
    articles = split_articles(text)
    nlp = _language_model()
    candidates = []

    for article in articles:
        body = article["text"]
        if not body:
            continue
        for sentence in nlp(body).sents:
            sentence_text = sentence.text.strip()
            markers = [
                name
                for name, pattern in REQUIREMENT_MARKERS.items()
                if pattern.search(sentence_text)
            ]
            if markers:
                candidates.append(
                    {
                        "article_number": article["number"],
                        "sentence": sentence_text,
                        "markers": markers,
                    }
                )

    return {
        "article_count": len(articles),
        "candidate_count": len(candidates),
        "articles": articles,
        "candidate_requirements": candidates,
        "warning": "Результаты получены по языковым маркерам и требуют проверки человеком.",
    }

