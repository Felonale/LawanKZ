"""Эксперимент первой недели второго рубежного контроля.

Модуль сравнивает контекстные Transformer-эмбеддинги с TF-IDF на одной
небольшой выборке статей нормативных правовых актов.  Результаты сохраняются
отдельно от материалов первого РК, чтобы исходный эксперимент не изменялся.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_REVISION = "b8ef00830037f9868450f778081ea683e900fe39"
DOCUMENT_IDS = (
    "personal_data",
    "cybersecurity",
    "communications",
    "artificial_intelligence",
    "access_to_information",
)
DOCUMENT_SHORT_NAMES = {
    "personal_data": "ПД",
    "cybersecurity": "КБ",
    "communications": "Связь",
    "artificial_intelligence": "ИИ",
    "access_to_information": "Доступ",
}


def load_articles(path: Path) -> list[dict]:
    """Прочитать подготовленные статьи из JSONL."""
    articles = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not articles:
        raise ValueError("Файл обработанных статей пуст")
    return articles


def select_balanced_articles(articles: list[dict]) -> list[dict]:
    """Выбрать 15 средних по длине статей: по три из пяти актов.

    Ограничение 80-250 очищенных слов уменьшает риск того, что результат будет
    определяться очень коротким определением или огромной составной статьёй.
    Из упорядоченного списка берутся короткий, медианный и длинный примеры.
    """
    groups: dict[str, list[dict]] = defaultdict(list)
    for article in articles:
        if article["document_id"] in DOCUMENT_IDS and 80 <= article["clean_word_count"] <= 250:
            groups[article["document_id"]].append(article)

    selected: list[dict] = []
    for document_id in DOCUMENT_IDS:
        candidates = sorted(
            groups[document_id],
            key=lambda item: (item["clean_word_count"], str(item["article_id"])),
        )
        if len(candidates) < 3:
            raise ValueError(f"Для {document_id} найдено меньше трёх подходящих статей")
        positions = (0, len(candidates) // 2, len(candidates) - 1)
        selected.extend(candidates[position] for position in positions)
    return selected


def split_into_chunks(text: str, *, max_words: int = 90) -> list[str]:
    """Разделить длинную статью на небольшие фрагменты без разрыва предложений."""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?;])\s+", text) if part.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current and current_words + sentence_words > max_words:
            chunks.append(" ".join(current))
            current = []
            current_words = 0
        current.append(sentence)
        current_words += sentence_words
    if current:
        chunks.append(" ".join(current))
    return chunks or [text]


def encode_articles(model, texts: list[str]) -> tuple[np.ndarray, list[int]]:
    """Получить нормированные векторы статей как среднее векторов фрагментов."""
    all_chunks: list[str] = []
    owners: list[int] = []
    chunk_counts: list[int] = []
    for index, text in enumerate(texts):
        chunks = split_into_chunks(text)
        chunk_counts.append(len(chunks))
        all_chunks.extend(chunks)
        owners.extend([index] * len(chunks))

    chunk_vectors = model.encode(
        all_chunks,
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    article_vectors = []
    for index in range(len(texts)):
        vector = chunk_vectors[np.asarray(owners) == index].mean(axis=0)
        norm = np.linalg.norm(vector)
        article_vectors.append(vector / norm if norm else vector)
    return np.asarray(article_vectors, dtype=np.float32), chunk_counts


def build_tfidf_vectors(texts: list[str]) -> tuple[np.ndarray, int]:
    """Построить TF-IDF по тем же текстам и вернуть плотную матрицу для сравнения."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[а-яёa-z][а-яёa-z-]+\b",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(texts)
    return normalize(matrix).toarray().astype(np.float32), len(vectorizer.vocabulary_)


def evaluate_similarity(vectors: np.ndarray, document_ids: list[str], *, top_k: int = 2) -> dict:
    """Оценить, поднимает ли представление статьи того же исходного акта."""
    similarities = cosine_similarity(vectors)
    same_scores: list[float] = []
    different_scores: list[float] = []
    precisions: list[float] = []
    hits: list[float] = []

    for left in range(len(document_ids)):
        for right in range(left + 1, len(document_ids)):
            target = same_scores if document_ids[left] == document_ids[right] else different_scores
            target.append(float(similarities[left, right]))
        order = np.argsort(-similarities[left])
        neighbours = [int(index) for index in order if index != left][:top_k]
        relevant = [document_ids[index] == document_ids[left] for index in neighbours]
        precisions.append(sum(relevant) / top_k)
        hits.append(float(any(relevant)))

    same_mean = float(np.mean(same_scores))
    different_mean = float(np.mean(different_scores))
    return {
        "same_document_mean_cosine": round(same_mean, 4),
        "different_document_mean_cosine": round(different_mean, 4),
        "cosine_separation": round(same_mean - different_mean, 4),
        f"precision_at_{top_k}": round(float(np.mean(precisions)), 4),
        f"hit_rate_at_{top_k}": round(float(np.mean(hits)), 4),
    }


def find_legal_operator_sentence(articles: list[dict]) -> tuple[str, str]:
    """Найти в корпусе компактное предложение со словом «оператор»."""
    preferred: list[tuple[str, str]] = []
    for article in articles:
        for sentence in re.split(r"(?<=[.!?])\s+", article["text"]):
            if (
                re.search(r"\bоператор\w*\b", sentence, flags=re.IGNORECASE)
                and "персональн" in sentence.lower()
                and 80 <= len(sentence) <= 360
                and "\n" not in sentence
            ):
                candidate = (article["article_id"], sentence.strip())
                if "оператором" in sentence.lower() and "обработ" in sentence.lower():
                    return candidate
                preferred.append(candidate)
    if preferred:
        return preferred[0]
    raise ValueError("В корпусе не найдено подходящее предложение со словом «оператор»")


def contextual_word_analysis(model, legal_sentence: str) -> tuple[list[dict], np.ndarray]:
    """Сравнить скрытые векторы слова «оператор» в трёх контекстах."""
    import torch

    contexts = [
        ("персональные данные", legal_sentence),
        ("телекоммуникации", "Мобильный оператор предоставляет абонентам услуги связи и доступ в интернет."),
        ("кинопроизводство", "Оператор снимает фильм на профессиональную камеру и выбирает композицию кадра."),
    ]
    transformer = model[0]
    tokenizer = transformer.tokenizer
    auto_model = transformer.auto_model
    auto_model.eval()
    vectors = []

    for _, sentence in contexts:
        match = re.search(r"оператор\w*", sentence, flags=re.IGNORECASE)
        if match is None:
            raise ValueError("Контекст не содержит слово «оператор»")
        encoded = tokenizer(
            sentence,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=model.max_seq_length,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            hidden = auto_model(**encoded).last_hidden_state[0]
        token_indices = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > match.start() and start < match.end()
        ]
        if not token_indices:
            raise ValueError("Не удалось сопоставить слово «оператор» с токенами модели")
        vector = hidden[token_indices].mean(dim=0).cpu().numpy()
        vectors.append(vector / np.linalg.norm(vector))

    matrix = cosine_similarity(np.asarray(vectors, dtype=np.float32))
    rows = [
        {"context": context, "sentence": sentence}
        for context, sentence in contexts
    ]
    return rows, matrix


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_comparison(
    tfidf_vectors: np.ndarray,
    transformer_vectors: np.ndarray,
    labels: list[str],
    metrics: dict[str, dict],
    output_path: Path,
) -> None:
    """Сохранить две матрицы сходства и компактное сравнение метрик."""
    tfidf_similarity = cosine_similarity(tfidf_vectors)
    transformer_similarity = cosine_similarity(transformer_vectors)
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(3.4, 1.2))
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    for axis, matrix, title in zip(
        axes,
        (tfidf_similarity, transformer_similarity),
        ("TF-IDF: косинусное сходство", "Transformer: косинусное сходство"),
    ):
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
        axis.set_title(title, fontsize=14, weight="bold")
        axis.set_xticks(range(len(labels)), labels, rotation=90, fontsize=8)
        axis.set_yticks(range(len(labels)), labels, fontsize=8)
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    axis = fig.add_subplot(grid[1, :])
    metric_names = ["cosine_separation", "precision_at_2", "hit_rate_at_2"]
    display_names = ["Разделение косинусов", "Precision@2", "Hit rate@2"]
    x = np.arange(len(metric_names))
    width = 0.34
    axis.bar(x - width / 2, [metrics["TF-IDF"][name] for name in metric_names], width, label="TF-IDF")
    axis.bar(
        x + width / 2,
        [metrics["Transformer"][name] for name in metric_names],
        width,
        label="Transformer",
    )
    axis.set_xticks(x, display_names)
    axis.set_ylim(0, 1.05)
    axis.grid(axis="y", alpha=0.2)
    axis.legend()
    axis.set_title("Сравнение представлений на 15 статьях", fontsize=14, weight="bold")
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def write_report(summary: dict, output_path: Path) -> None:
    """Сформировать краткий текстовый отчёт по требованиям первой недели."""
    tfidf = summary["metrics"]["TF-IDF"]
    transformer = summary["metrics"]["Transformer"]
    context = summary["context_analysis"]
    report = f"""# Неделя 1. Transformer и механизм Attention

## Цель и данные

Эксперимент проверяет, чем контекстные представления Transformer отличаются от
TF-IDF на материалах проекта LawanKZ. Из корпуса выбраны 15 статей: по три статьи
из пяти нормативных правовых актов. Для длинных статей Transformer-эмбеддинг
вычисляется по фрагментам примерно по 90 слов, после чего векторы усредняются и
нормируются. Использована модель `{summary['model_name']}`.

## Как работает Attention

Transformer рассматривает каждый токен не изолированно. Для токена строятся
векторы запроса Q, ключа K и значения V. Скалярное произведение Q и K показывает,
каким соседним токенам следует уделить больше внимания; после softmax получаются
веса, которыми суммируются V. Несколько attention-heads позволяют одновременно
учитывать разные связи: например, термин, объект действия и модальную конструкцию.

## Сравнение с TF-IDF

| Представление | Средний косинус внутри одного акта | Между актами | Разделение | Precision@2 | Hit rate@2 |
|---|---:|---:|---:|---:|---:|
| TF-IDF | {tfidf['same_document_mean_cosine']:.4f} | {tfidf['different_document_mean_cosine']:.4f} | {tfidf['cosine_separation']:.4f} | {tfidf['precision_at_2']:.4f} | {tfidf['hit_rate_at_2']:.4f} |
| Transformer | {transformer['same_document_mean_cosine']:.4f} | {transformer['different_document_mean_cosine']:.4f} | {transformer['cosine_separation']:.4f} | {transformer['precision_at_2']:.4f} | {transformer['hit_rate_at_2']:.4f} |

В этой учебной оценке релевантной считается статья из того же исходного акта.
Метрика удобна для воспроизводимого сравнения, но не заменяет экспертную оценку
юриста: разные акты могут регулировать близкие вопросы, а статьи одного акта —
описывать разные процедуры.

На выбранной малой и однородной выборке TF-IDF оказался сильнее по Precision@2:
{tfidf['precision_at_2']:.4f} против {transformer['precision_at_2']:.4f}. Это не
опровержение Transformer как класса моделей. Общая модель формирует для
канцелярски похожих статей очень близкие векторы, а TF-IDF лучше использует
редкие названия предметной области. Для юридического поиска результат указывает
на необходимость доменной адаптации, более крупной оценки и сравнения других
моделей.

## Контекст слова «оператор»

В корпусе разобрано предложение из `{context['source_article_id']}`:

> {context['sentences'][0]['sentence']}

Здесь «оператор» — участник обработки персональных данных. В предложении об
услугах связи слово обозначает телекоммуникационную организацию, а в предложении
о съёмке фильма — специалиста за камерой. Модель создаёт три разных скрытых
вектора одного написания слова. Косинус между юридическим и телекоммуникационным
контекстами равен {context['legal_vs_telecom_cosine']:.4f}, а между юридическим и
кинематографическим — {context['legal_vs_cinema_cosine']:.4f}.
В данном запуске второй косинус оказался выше первого, поэтому три примера нельзя
считать полноценной проверкой различения значений: они демонстрируют зависимость
вектора от окружения, но не доказывают соответствие геометрии человеческому
толкованию слова.

## Вывод

TF-IDF прозрачен: координата соответствует конкретному слову или биграмме, но
порядок и контекст учитываются слабо. Transformer формирует плотный вектор всего
фрагмента и меняет представление слова в зависимости от окружения. Поэтому он
может сближать тексты без полного совпадения слов. Цена этого преимущества —
большая вычислительная стоимость, ограничение длины входа и менее наглядная
интерпретация. Результат на 15 статьях является демонстрационным, а не итоговой
оценкой качества юридического поиска.
"""
    output_path.write_text(report, encoding="utf-8")


def run_week1_analysis(input_path: Path, output_dir: Path) -> dict:
    """Запустить полный эксперимент и сохранить все результаты первой недели."""
    from sentence_transformers import SentenceTransformer

    output_dir.mkdir(parents=True, exist_ok=True)
    articles = load_articles(input_path)
    selected = select_balanced_articles(articles)
    texts = [article["text"] for article in selected]
    document_ids = [article["document_id"] for article in selected]

    model = SentenceTransformer(MODEL_NAME, revision=MODEL_REVISION, device="cpu")
    transformer_vectors, chunk_counts = encode_articles(model, texts)
    tfidf_vectors, vocabulary_size = build_tfidf_vectors(texts)

    metrics = {
        "TF-IDF": evaluate_similarity(tfidf_vectors, document_ids),
        "Transformer": evaluate_similarity(transformer_vectors, document_ids),
    }
    selected_rows = []
    labels = []
    for index, (article, chunk_count) in enumerate(zip(selected, chunk_counts), start=1):
        label = f"{DOCUMENT_SHORT_NAMES[article['document_id']]}-{article['article_number']}"
        labels.append(label)
        selected_rows.append(
            {
                "sample_id": index,
                "label": label,
                "article_id": article["article_id"],
                "document_id": article["document_id"],
                "document_title": article["document_title"],
                "article_number": article["article_number"],
                "clean_word_count": article["clean_word_count"],
                "transformer_chunks": chunk_count,
                "preview": " ".join(article["text"].split())[:240],
            }
        )
    _write_csv(
        output_dir / "selected_texts.csv",
        selected_rows,
        list(selected_rows[0]),
    )
    np.save(output_dir / "transformer_embeddings.npy", transformer_vectors)
    np.save(output_dir / "tfidf_embeddings.npy", tfidf_vectors)

    source_article_id, legal_sentence = find_legal_operator_sentence(articles)
    context_rows, context_matrix = contextual_word_analysis(model, legal_sentence)
    context_summary = {
        "target_word": "оператор",
        "source_article_id": source_article_id,
        "sentences": context_rows,
        "legal_vs_telecom_cosine": round(float(context_matrix[0, 1]), 4),
        "legal_vs_cinema_cosine": round(float(context_matrix[0, 2]), 4),
        "telecom_vs_cinema_cosine": round(float(context_matrix[1, 2]), 4),
    }
    (output_dir / "context_analysis.json").write_text(
        json.dumps(context_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    plot_comparison(
        tfidf_vectors,
        transformer_vectors,
        labels,
        metrics,
        output_dir / "tfidf_vs_transformer.png",
    )
    summary = {
        "week": 1,
        "selected_text_count": len(selected),
        "document_count": len(set(document_ids)),
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "transformer_dimension": int(transformer_vectors.shape[1]),
        "tfidf_dimension": int(tfidf_vectors.shape[1]),
        "tfidf_vocabulary_size": vocabulary_size,
        "metrics": metrics,
        "context_analysis": context_summary,
        "evaluation_note": "Релевантность приближённо определяется принадлежностью статьи к тому же акту.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_report(summary, output_dir / "report.md")
    return summary

