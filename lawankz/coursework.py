"""Воспроизводимый анализ корпуса по заданиям недель 2–4."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import spacy
from matplotlib import font_manager
from nltk.stem.snowball import SnowballStemmer
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from spacy.lang.ru.stop_words import STOP_WORDS
from wordcloud import WordCloud

from .preprocessing import split_articles


CYRILLIC_WORD = re.compile(r"^[а-яё]+(?:-[а-яё]+)*$", re.IGNORECASE)
# Общий список spaCy считает некоторые юридически важные слова стоп-словами.
KEEP_WORDS = {
    "не", "без", "нет", "нельзя", "данные", "должен", "должна", "должны",
    "должно", "должный", "обязан", "обязана", "обязаны", "вправе",
    "допускается", "запрещается", "подлежит",
}


def clean_token(token) -> tuple[str, str] | None:
    """Нормализация токена"""
    word = token.text.lower().strip()
    lemma = token.lemma_.lower().strip() if token.lemma_ else word
    if not CYRILLIC_WORD.fullmatch(word) or len(word) < 2:
        return None
    if (word in STOP_WORDS or lemma in STOP_WORDS) and not (
        word in KEEP_WORDS or lemma in KEEP_WORDS
    ):
        return None
    if not CYRILLIC_WORD.fullmatch(lemma):
        lemma = word
    return word, lemma


def preprocess_text(text: str, nlp=None) -> dict[str, list[str]]:
    """токенизация, очистка, леммы и стемы."""
    if nlp is None:
        nlp = spacy.load("ru_core_news_sm", disable=["ner", "parser"])
    stemmer = SnowballStemmer("russian")
    words = []
    lemmas = []
    stems = []
    for token in nlp(text):
        cleaned = clean_token(token)
        if cleaned is None:
            continue
        word, lemma = cleaned
        words.append(word)
        lemmas.append(lemma)
        stems.append(stemmer.stem(word))
    return {"words": words, "lemmas": lemmas, "stems": stems}


def prepare_articles(documents: list[dict]) -> list[dict]:
    """Превратить документы в строки для сравнения статей."""
    articles = []
    for document in documents:
        for index, article in enumerate(split_articles(document["text"]), start=1):
            text = article["text"] or ""
            if len(text.split()) < 10:
                continue
            articles.append({
                "article_id": f"{document['id']}:{index}",
                "document_id": document["id"],
                "document_title": document["title"],
                "article_number": article["number"],
                "article_title": article["title"],
                "text": text,
            })
    return articles


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_top_terms(bow_terms: list[dict], tfidf_terms: list[dict], output_path: Path) -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    for ax, rows, value_key, title, color in (
        (axes[0], bow_terms[:15], "score", "BoW: сумма употреблений", "#28645f"),
        (axes[1], tfidf_terms[:15], "score", "TF-IDF: средний вес по статьям", "#aa6a34"),
    ):
        terms = [row["term"] for row in rows][::-1]
        values = [row[value_key] for row in rows][::-1]
        ax.barh(terms, values, color=color)
        ax.set_title(title, fontsize=15)
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("Частые слова и слова с высоким TF-IDF", fontsize=18)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_wordcloud(lemma_counts: Counter[str], output_path: Path) -> None:
    font_path = font_manager.findfont("DejaVu Sans")
    cloud = WordCloud(
        font_path=font_path,
        width=1600,
        height=900,
        background_color="white",
        colormap="viridis",
        max_words=100,
        random_state=42,
    ).generate_from_frequencies(lemma_counts)
    cloud.to_file(str(output_path))


def run_analysis(documents: list[dict], output_dir: Path) -> dict:
    """Недели 2–4: очистка, морфология, частоты, BoW и TF-IDF."""
    output_dir.mkdir(parents=True, exist_ok=True)
    articles = prepare_articles(documents)
    if not articles:
        raise ValueError("Нет статей для анализа")

    nlp = spacy.load("ru_core_news_sm", disable=["ner", "parser"])
    stemmer = SnowballStemmer("russian")
    word_counts: Counter[str] = Counter()
    lemma_counts: Counter[str] = Counter()
    morphology_counts: Counter[tuple[str, str, str]] = Counter()
    comparison_counts: Counter[tuple[str, str, str, str, str, str]] = Counter()
    total_tokens_before = 0
    total_tokens_after = 0

    for article, doc in zip(articles, nlp.pipe((row["text"] for row in articles), batch_size=16)):
        words = []
        lemmas = []
        stems = []
        for token in doc:
            if CYRILLIC_WORD.fullmatch(token.text):
                total_tokens_before += 1
            cleaned = clean_token(token)
            if cleaned is None:
                continue
            word, lemma = cleaned
            stem = stemmer.stem(word)
            pos = token.pos_ or "UNKNOWN"
            case = "/".join(token.morph.get("Case")) or "—"
            number = "/".join(token.morph.get("Number")) or "—"
            words.append(word)
            lemmas.append(lemma)
            stems.append(stem)
            word_counts[word] += 1
            lemma_counts[lemma] += 1
            morphology_counts[(pos, case, number)] += 1
            comparison_counts[(word, lemma, stem, pos, case, number)] += 1
            total_tokens_after += 1
        article["clean_words"] = words
        article["lemmas"] = lemmas
        article["stems"] = stems
        article["clean_word_count"] = len(words)

    cleaned_articles = [article for article in articles if article["clean_word_count"] >= 5]
    texts = [" ".join(article["lemmas"]) for article in cleaned_articles]
    vectorizer = CountVectorizer(min_df=2, max_df=0.95)
    bow = vectorizer.fit_transform(texts)
    tfidf = TfidfTransformer().fit_transform(bow)
    features = vectorizer.get_feature_names_out()
    bow_sums = np.asarray(bow.sum(axis=0)).ravel()
    tfidf_means = np.asarray(tfidf.mean(axis=0)).ravel()

    bow_ranked = np.argsort(-bow_sums)
    tfidf_ranked = np.argsort(-tfidf_means)
    top_bow = [{"term": str(features[i]), "score": int(bow_sums[i])} for i in bow_ranked[:50]]
    top_tfidf = [{"term": str(features[i]), "score": round(float(tfidf_means[i]), 6)} for i in tfidf_ranked[:50]]

    distinctive_terms = []
    document_ids = sorted({article["document_id"] for article in cleaned_articles})
    for document_id in document_ids:
        own_rows = [i for i, article in enumerate(cleaned_articles) if article["document_id"] == document_id]
        other_rows = [i for i, article in enumerate(cleaned_articles) if article["document_id"] != document_id]
        own_mean = np.asarray(tfidf[own_rows].mean(axis=0)).ravel()
        other_mean = np.asarray(tfidf[other_rows].mean(axis=0)).ravel()
        differences = own_mean - other_mean
        for i in np.argsort(-differences)[:10]:
            distinctive_terms.append({
                "document_id": document_id,
                "term": str(features[i]),
                "own_mean_tfidf": round(float(own_mean[i]), 6),
                "other_mean_tfidf": round(float(other_mean[i]), 6),
                "difference": round(float(differences[i]), 6),
            })

    _write_csv(
        output_dir / "top50_words.csv",
        [{"word": word, "frequency": frequency} for word, frequency in word_counts.most_common(50)],
        ["word", "frequency"],
    )
    _write_csv(
        output_dir / "top50_lemmas.csv",
        [{"lemma": lemma, "frequency": frequency} for lemma, frequency in lemma_counts.most_common(50)],
        ["lemma", "frequency"],
    )
    _write_csv(
        output_dir / "morphology.csv",
        [
            {"pos": pos, "case": case, "number": number, "frequency": frequency}
            for (pos, case, number), frequency in morphology_counts.most_common()
        ],
        ["pos", "case", "number", "frequency"],
    )
    comparison_rows = [
        {"word": word, "lemma_spacy": lemma, "stem_snowball": stem, "pos": pos,
         "case": case, "number": number, "frequency": frequency}
        for (word, lemma, stem, pos, case, number), frequency in comparison_counts.most_common(100)
    ]
    _write_csv(
        output_dir / "lemma_vs_stem.csv", comparison_rows,
        ["word", "lemma_spacy", "stem_snowball", "pos", "case", "number", "frequency"],
    )
    _write_csv(output_dir / "top50_bow.csv", top_bow, ["term", "score"])
    _write_csv(output_dir / "top50_tfidf.csv", top_tfidf, ["term", "score"])
    _write_csv(
        output_dir / "distinctive_terms_by_document.csv", distinctive_terms,
        ["document_id", "term", "own_mean_tfidf", "other_mean_tfidf", "difference"],
    )
    _write_csv(
        output_dir / "article_index.csv",
        [
            {key: article[key] for key in ("article_id", "document_id", "document_title", "article_number", "article_title", "clean_word_count")}
            for article in cleaned_articles
        ],
        ["article_id", "document_id", "document_title", "article_number", "article_title", "clean_word_count"],
    )
    (output_dir / "processed_articles.jsonl").write_text(
        "".join(json.dumps(article, ensure_ascii=False) + "\n" for article in cleaned_articles),
        encoding="utf-8",
    )
    save_npz(output_dir / "bow_matrix.npz", bow)
    save_npz(output_dir / "tfidf_matrix.npz", tfidf)
    _write_csv(
        output_dir / "vocabulary.csv",
        [{"feature_index": i, "term": term} for i, term in enumerate(features)],
        ["feature_index", "term"],
    )
    _plot_top_terms(top_bow, top_tfidf, output_dir / "bow_vs_tfidf.png")
    _plot_wordcloud(lemma_counts, output_dir / "wordcloud_lemmas.png")

    summary = {
        "document_count": len(documents),
        "raw_word_count": sum(document["word_count"] for document in documents),
        "article_count": len(cleaned_articles),
        "tokens_before_cleaning": total_tokens_before,
        "tokens_after_cleaning": total_tokens_after,
        "unique_words": len(word_counts),
        "unique_lemmas": len(lemma_counts),
        "bow_shape": list(bow.shape),
        "tfidf_shape": list(tfidf.shape),
        "top_bow": top_bow[:15],
        "top_tfidf": top_tfidf[:15],
        "distinctive_terms_by_document": {
            document_id: [row["term"] for row in distinctive_terms if row["document_id"] == document_id][:5]
            for document_id in document_ids
        },
        "preprocessing_example": {
            "before": cleaned_articles[0]["text"][:300],
            "after_words": cleaned_articles[0]["clean_words"][:35],
            "after_lemmas": cleaned_articles[0]["lemmas"][:35],
        },
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
