"""Воспроизводимый небольшой корпус НПА из официальной ИПС «Әділет»."""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup


WORD_PATTERN = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")
SKIP_PREFIXES = (
    "Сноска.",
    "Примечание ИЗПИ!",
    "Примечание РЦПИ!",
    "Вниманию пользователей!",
    "Для удобства пользования",
)
USER_AGENT = "LawanKZ-university-NLP-project/0.1 (7 public documents; cached snapshots)"


def load_sources(path: Path) -> list[dict[str, str]]:
    sources = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(sources, list) or not sources:
        raise ValueError("Файл источников должен содержать непустой список")
    ids = set()
    for source in sources:
        if not isinstance(source, dict) or not all(
            isinstance(source.get(key), str) and source[key]
            for key in ("id", "title", "topic", "url")
        ):
            raise ValueError("У каждого источника нужны id, title, topic и url")
        if source["id"] in ids:
            raise ValueError(f"Повторяющийся id: {source['id']}")
        ids.add(source["id"])
        parsed = urlparse(source["url"])
        if parsed.scheme != "https" or parsed.netloc != "old.adilet.zan.kz" or not parsed.path.startswith("/rus/docs/"):
            raise ValueError(f"Неожиданный адрес источника: {source['url']}")
    return sources


def extract_act_text(html: str) -> str:
    """Взять основное тело акта, исключив меню сайта и редакционные примечания."""
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one(".text_upd article")
    if article is None:
        raise ValueError("Не найдено тело акта (.text_upd article): возможно, сайт изменился")

    for note in article.select(".note, script, style"):
        note.decompose()

    blocks = []
    for tag in article.find_all(("p", "h1", "h2", "h3", "h4", "li")):
        if tag.find_parent(("p", "li")) is not None:
            continue
        line = re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()
        if line and not line.startswith(SKIP_PREFIXES):
            blocks.append(line)

    text = "\n".join(blocks)
    if len(WORD_PATTERN.findall(text)) < 1000:
        raise ValueError("После извлечения осталось меньше 1000 русских слов: проверьте HTML")
    return text


def fetch_snapshots(sources: list[dict[str, str]], raw_dir: Path, refresh: bool = False) -> None:
    """Один раз скачать указанные страницы; далее использовать сохранённые копии."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    needed = [
        source for source in sources
        if refresh
        or not (raw_dir / f"{source['id']}.html").exists()
        or not (raw_dir / f"{source['id']}.meta.json").exists()
    ]
    if not needed:
        return

    with httpx.Client(timeout=40, follow_redirects=True, headers={"User-Agent": USER_AGENT}) as client:
        robots_response = client.get("https://old.adilet.zan.kz/robots.txt")
        robots_response.raise_for_status()
        robots = RobotFileParser()
        robots.parse(robots_response.text.splitlines())

        for index, source in enumerate(needed):
            if not robots.can_fetch(USER_AGENT, source["url"]):
                raise PermissionError(f"robots.txt запрещает загрузку: {source['url']}")
            if index:
                time.sleep(1.2)
            response = client.get(source["url"])
            response.raise_for_status()
            if "text/html" not in response.headers.get("content-type", ""):
                raise ValueError(f"Источник вернул не HTML: {source['url']}")
            html = response.content
            extract_act_text(html.decode("utf-8"))  # проверка до сохранения снимка
            (raw_dir / f"{source['id']}.html").write_bytes(html)
            metadata = {
                "url": source["url"],
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "sha256": hashlib.sha256(html).hexdigest(),
            }
            (raw_dir / f"{source['id']}.meta.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )


def build_documents(sources: list[dict[str, str]], raw_dir: Path) -> list[dict]:
    documents = []
    for source in sources:
        html_path = raw_dir / f"{source['id']}.html"
        meta_path = raw_dir / f"{source['id']}.meta.json"
        raw_html = html_path.read_bytes()
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        sha256 = hashlib.sha256(raw_html).hexdigest()
        if sha256 != metadata["sha256"]:
            raise ValueError(f"Контрольная сумма снимка не совпала: {html_path}")
        text = extract_act_text(raw_html.decode("utf-8"))
        documents.append({
            **source,
            "fetched_at_utc": metadata["fetched_at_utc"],
            "sha256": sha256,
            "word_count": len(WORD_PATTERN.findall(text)),
            "text": text,
        })
    return documents


def save_documents(documents: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(document, ensure_ascii=False) + "\n" for document in documents),
        encoding="utf-8",
    )


def load_documents(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
