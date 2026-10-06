"""Векторные модели и поиск похожих статей для недель 5–6."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gensim.models import FastText, KeyedVectors, Word2Vec
from scipy.sparse import load_npz
from sklearn.decomposition import PCA


SEED = 42
EVALUATION_GROUPS = {
    "персональные данные": {"персональный", "данные", "субъект", "согласие", "сбор", "обработка"},
    "искусственный интеллект": {"искусственный", "интеллект", "система", "модель", "алгоритм", "риск"},
    "связь": {"связь", "сеть", "оператор", "абонент", "услуга", "устройство"},
    "кибербезопасность": {"кибербезопасности", "угроза", "безопасность", "инцидент", "защита"},
    "цифровая среда": {"цифровой", "объект", "среда", "электронный", "подпись"},
}
ASSOCIATION_PAIRS = [
    ("персональный", "данные"),
    ("искусственный", "интеллект"),
    ("электронный", "подпись"),
    ("оператор", "связь"),
    ("субъект", "согласие"),
    ("угроза", "безопасность"),
]


def load_processed_articles(path: Path) -> list[dict]:
    """Загрузить JSONL, созданный анализом недель 1–4."""
    articles = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                articles.append(json.loads(line))
    if not articles:
        raise ValueError("Файл обработанных статей пуст")
    return articles


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_glove(
    sentences: list[list[str]],
    *,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 3,
    epochs: int = 35,
    learning_rate: float = 0.05,
) -> tuple[KeyedVectors, list[float]]:
    """Обучить компактный GloVe по взвешенной матрице совместной встречаемости.

    Это учебная реализация исходной целевой функции GloVe, а не готовые
    англоязычные векторы. Контекст ближе к слову получает вес 1 / distance.
    """
    counts = Counter(word for sentence in sentences for word in sentence)
    words = sorted(word for word, count in counts.items() if count >= min_count)
    word_to_id = {word: index for index, word in enumerate(words)}
    cooccurrence: defaultdict[tuple[int, int], float] = defaultdict(float)

    for sentence in sentences:
        ids = [word_to_id[word] for word in sentence if word in word_to_id]
        for center_position, center_id in enumerate(ids):
            left = max(0, center_position - window)
            right = min(len(ids), center_position + window + 1)
            for context_position in range(left, right):
                if context_position == center_position:
                    continue
                distance = abs(context_position - center_position)
                cooccurrence[(center_id, ids[context_position])] += 1.0 / distance

    if not cooccurrence:
        raise ValueError("Недостаточно совместных вхождений для GloVe")

    pairs = np.asarray(list(cooccurrence.keys()), dtype=np.int32)
    values = np.asarray(list(cooccurrence.values()), dtype=np.float32)
    rng = np.random.default_rng(SEED)
    scale = 0.5 / vector_size
    word_vectors = rng.normal(0, scale, (len(words), vector_size)).astype(np.float32)
    context_vectors = rng.normal(0, scale, (len(words), vector_size)).astype(np.float32)
    word_bias = np.zeros(len(words), dtype=np.float32)
    context_bias = np.zeros(len(words), dtype=np.float32)
    grad_word = np.ones_like(word_vectors)
    grad_context = np.ones_like(context_vectors)
    grad_word_bias = np.ones(len(words), dtype=np.float32)
    grad_context_bias = np.ones(len(words), dtype=np.float32)
    weights = np.minimum(1.0, (values / 100.0) ** 0.75).astype(np.float32)
    targets = np.log(values).astype(np.float32)
    losses = []

    order = np.arange(len(values))
    for _ in range(epochs):
        rng.shuffle(order)
        epoch_loss = 0.0
        for position in order:
            i, j = pairs[position]
            wi = word_vectors[i].copy()
            cj = context_vectors[j].copy()
            error = float(np.dot(wi, cj) + word_bias[i] + context_bias[j] - targets[position])
            weighted_error = float(weights[position] * error)
            epoch_loss += 0.5 * weights[position] * error * error

            gradient_w = weighted_error * cj
            gradient_c = weighted_error * wi
            word_vectors[i] -= learning_rate * gradient_w / np.sqrt(grad_word[i])
            context_vectors[j] -= learning_rate * gradient_c / np.sqrt(grad_context[j])
            word_bias[i] -= learning_rate * weighted_error / math.sqrt(float(grad_word_bias[i]))
            context_bias[j] -= learning_rate * weighted_error / math.sqrt(float(grad_context_bias[j]))
            grad_word[i] += gradient_w * gradient_w
            grad_context[j] += gradient_c * gradient_c
            grad_word_bias[i] += weighted_error * weighted_error
            grad_context_bias[j] += weighted_error * weighted_error
        losses.append(float(epoch_loss / len(values)))

    keyed_vectors = KeyedVectors(vector_size=vector_size)
    keyed_vectors.add_vectors(words, word_vectors + context_vectors)
    return keyed_vectors, losses


def train_embedding_models(
    articles: list[dict], output_dir: Path, *, vector_size: int = 100
) -> dict[str, KeyedVectors]:
    """Обучить и сохранить Word2Vec, fastText и GloVe на одних данных."""
    output_dir.mkdir(parents=True, exist_ok=True)
    sentences = [article["lemmas"] for article in articles if len(article["lemmas"]) >= 5]
    common = dict(
        sentences=sentences,
        vector_size=vector_size,
        window=5,
        min_count=3,
        workers=1,
        epochs=80,
        seed=SEED,
    )
    word2vec = Word2Vec(sg=1, negative=10, **common)
    # Стандартные 2 млн корзин создают файл примерно 800 МБ даже для нашего
    # маленького корпуса. 50 тыс. достаточно для учебной выборки и репозитория.
    fasttext = FastText(
        sg=1, negative=10, min_n=3, max_n=6, bucket=50_000, **common
    )
    glove, glove_losses = train_glove(
        sentences, vector_size=vector_size, window=5, min_count=3
    )

    word2vec.save(str(output_dir / "word2vec.model"))
    fasttext.save(str(output_dir / "fasttext.model"))
    glove.save(str(output_dir / "glove.kv"))
    (output_dir / "glove_training.json").write_text(
        json.dumps(
            {
                "epochs": len(glove_losses),
                "initial_loss": glove_losses[0],
                "final_loss": glove_losses[-1],
                "losses": glove_losses,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"Word2Vec": word2vec.wv, "fastText": fasttext.wv, "GloVe": glove}


def _contains(model: KeyedVectors, word: str, *, allow_subword: bool = False) -> bool:
    if allow_subword:
        try:
            model.get_vector(word)
            return True
        except KeyError:
            return False
    return word in model.key_to_index


def evaluate_models(models: dict[str, KeyedVectors], output_dir: Path) -> list[dict]:
    """Сравнить покрытие, тематические соседства и известные пары."""
    rows = []
    all_terms = sorted(set().union(*EVALUATION_GROUPS.values()))
    for name, model in models.items():
        subword = name == "fastText"
        in_vocab = [word for word in all_terms if word in model.key_to_index]
        pair_scores = [
            float(model.similarity(left, right))
            for left, right in ASSOCIATION_PAIRS
            if _contains(model, left, allow_subword=subword)
            and _contains(model, right, allow_subword=subword)
        ]
        precisions = []
        for group in EVALUATION_GROUPS.values():
            for seed in sorted(group):
                if seed not in model.key_to_index:
                    continue
                neighbours = [word for word, _ in model.most_similar(seed, topn=5)]
                precisions.append(sum(word in group - {seed} for word in neighbours) / 5)
        rows.append(
            {
                "model": name,
                "vocabulary_size": len(model.key_to_index),
                "evaluation_coverage": round(len(in_vocab) / len(all_terms), 3),
                "mean_association_cosine": round(float(np.mean(pair_scores)), 4),
                "mean_group_precision_at_5": round(float(np.mean(precisions)), 4),
                "oov_typo_supported": _contains(model, "персональнвй", allow_subword=subword),
            }
        )
    _write_csv(
        output_dir / "model_comparison.csv",
        rows,
        [
            "model",
            "vocabulary_size",
            "evaluation_coverage",
            "mean_association_cosine",
            "mean_group_precision_at_5",
            "oov_typo_supported",
        ],
    )
    return rows


def save_semantic_examples(models: dict[str, KeyedVectors], output_dir: Path) -> list[dict]:
    """Сохранить ближайшие слова и осторожные примеры векторной арифметики."""
    query_words = ["данные", "связь", "интеллект", "кибербезопасности", "согласие", "подпись"]
    neighbour_rows = []
    for name, model in models.items():
        for query in query_words:
            if query not in model.key_to_index:
                continue
            for rank, (word, similarity) in enumerate(model.most_similar(query, topn=8), start=1):
                neighbour_rows.append(
                    {"model": name, "query": query, "rank": rank, "word": word, "cosine": round(float(similarity), 4)}
                )
    _write_csv(
        output_dir / "nearest_words.csv",
        neighbour_rows,
        ["model", "query", "rank", "word", "cosine"],
    )

    equations = [
        ("персональный - данные + информация", ["персональный", "информация"], ["данные"]),
        ("оператор - связь + данные", ["оператор", "данные"], ["связь"]),
        ("цифровой - среда + система", ["цифровой", "система"], ["среда"]),
    ]
    arithmetic_rows = []
    for name, model in models.items():
        for equation, positive, negative in equations:
            if not all(word in model.key_to_index for word in positive + negative):
                continue
            for rank, (word, similarity) in enumerate(
                model.most_similar(positive=positive, negative=negative, topn=5), start=1
            ):
                arithmetic_rows.append(
                    {"model": name, "equation": equation, "rank": rank, "word": word, "cosine": round(float(similarity), 4)}
                )
    _write_csv(
        output_dir / "vector_arithmetic.csv",
        arithmetic_rows,
        ["model", "equation", "rank", "word", "cosine"],
    )
    return neighbour_rows


def build_idf(results_1_4_dir: Path) -> dict[str, float]:
    """Восстановить IDF из сохранённых BoW и словаря."""
    bow = load_npz(results_1_4_dir / "bow_matrix.npz")
    vocabulary = []
    with (results_1_4_dir / "vocabulary.csv").open(encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            vocabulary.append(row["term"])
    document_frequency = np.asarray((bow > 0).sum(axis=0)).ravel()
    values = np.log((1 + bow.shape[0]) / (1 + document_frequency)) + 1
    return {term: float(values[index]) for index, term in enumerate(vocabulary)}


def weighted_mean_vector(
    words: list[str], model: KeyedVectors, idf: dict[str, float]
) -> np.ndarray | None:
    """Представить текст средним вектором слов с весами IDF."""
    vectors = []
    weights = []
    for word in words:
        if word not in model.key_to_index:
            continue
        vectors.append(model.get_vector(word))
        weights.append(idf.get(word, 1.0))
    if not vectors:
        return None
    vector = np.average(np.asarray(vectors), axis=0, weights=np.asarray(weights))
    norm = np.linalg.norm(vector)
    return vector / norm if norm else None


def build_article_vectors(
    articles: list[dict], model: KeyedVectors, idf: dict[str, float]
) -> tuple[list[dict], np.ndarray]:
    """Построить нормированные векторы статей и удалить пустые строки."""
    kept_articles = []
    vectors = []
    for article in articles:
        vector = weighted_mean_vector(article["lemmas"], model, idf)
        if vector is not None:
            kept_articles.append(article)
            vectors.append(vector)
    return kept_articles, np.asarray(vectors, dtype=np.float32)


def search_by_vector(
    query_words: list[str],
    model: KeyedVectors,
    idf: dict[str, float],
    articles: list[dict],
    article_vectors: np.ndarray,
    *,
    top_k: int = 5,
) -> list[dict]:
    """Найти статьи по косинусу между запросом и векторами статей."""
    query_vector = weighted_mean_vector(query_words, model, idf)
    if query_vector is None:
        raise ValueError("В запросе нет слов, известных модели")
    similarities = article_vectors @ query_vector
    results = []
    for index in np.argsort(-similarities)[:top_k]:
        article = articles[int(index)]
        results.append(
            {
                "article_id": article["article_id"],
                "document_id": article["document_id"],
                "document_title": article["document_title"],
                "article_number": article["article_number"],
                "article_title": article["article_title"],
                "similarity": round(float(similarities[index]), 4),
                "preview": " ".join(article["text"].split())[:240],
            }
        )
    return results


def evaluate_article_retrieval(articles: list[dict], vectors: np.ndarray, *, top_k: int = 5) -> dict:
    """Leave-one-out: доля соседей статьи из того же исходного акта."""
    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, -np.inf)
    precisions = []
    hits = []
    for index, article in enumerate(articles):
        neighbours = np.argsort(-similarities[index])[:top_k]
        relevant = [articles[int(other)]["document_id"] == article["document_id"] for other in neighbours]
        precisions.append(sum(relevant) / top_k)
        hits.append(any(relevant))
    return {
        "article_count": len(articles),
        "top_k": top_k,
        "precision_at_5": round(float(np.mean(precisions)), 4),
        "hit_rate_at_5": round(float(np.mean(hits)), 4),
        "evaluation_note": "Запросом служит сама статья; она исключена, релевантны статьи того же акта.",
    }


def plot_article_pca(articles: list[dict], vectors: np.ndarray, output_path: Path) -> list[dict]:
    """Снизить размерность до 2D и сохранить диаграмму и координаты."""
    coordinates = PCA(n_components=2, random_state=SEED).fit_transform(vectors)
    document_ids = sorted({article["document_id"] for article in articles})
    palette = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(13, 8), constrained_layout=True)
    rows = []
    for color_index, document_id in enumerate(document_ids):
        indices = [i for i, article in enumerate(articles) if article["document_id"] == document_id]
        title = articles[indices[0]]["document_title"]
        ax.scatter(
            coordinates[indices, 0], coordinates[indices, 1],
            s=30, alpha=0.68, color=palette(color_index), label=title,
        )
        centroid = coordinates[indices].mean(axis=0)
        ax.annotate(document_id, centroid, fontsize=9, weight="bold")
        for index in indices:
            rows.append(
                {
                    "article_id": articles[index]["article_id"],
                    "document_id": document_id,
                    "pca_1": round(float(coordinates[index, 0]), 6),
                    "pca_2": round(float(coordinates[index, 1]), 6),
                }
            )
    ax.set_title("PCA-проекция векторов статей (Word2Vec + IDF)", fontsize=16)
    ax.set_xlabel("Первая главная компонента")
    ax.set_ylabel("Вторая главная компонента")
    ax.grid(alpha=0.18)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return rows


def run_embeddings_analysis(results_1_4_dir: Path, output_dir: Path) -> dict:
    """Полный воспроизводимый прогон недель 5–6."""
    output_dir.mkdir(parents=True, exist_ok=True)
    articles = load_processed_articles(results_1_4_dir / "processed_articles.jsonl")
    models = train_embedding_models(articles, output_dir)
    comparison = evaluate_models(models, output_dir)
    save_semantic_examples(models, output_dir)

    idf = build_idf(results_1_4_dir)
    (output_dir / "idf.json").write_text(
        json.dumps(idf, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    search_articles, article_vectors = build_article_vectors(articles, models["Word2Vec"], idf)
    np.save(output_dir / "article_vectors.npy", article_vectors)
    (output_dir / "search_articles.jsonl").write_text(
        "".join(json.dumps(article, ensure_ascii=False) + "\n" for article in search_articles),
        encoding="utf-8",
    )
    pca_rows = plot_article_pca(search_articles, article_vectors, output_dir / "article_vectors_pca.png")
    _write_csv(output_dir / "article_vectors_pca.csv", pca_rows, ["article_id", "document_id", "pca_1", "pca_2"])

    retrieval = evaluate_article_retrieval(search_articles, article_vectors)
    (output_dir / "retrieval_metrics.json").write_text(
        json.dumps(retrieval, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    sample_queries = [
        "согласие на обработку персональных данных",
        "защита от угроз кибербезопасности",
        "оператор сети связи оказывает услуги абоненту",
        "риски системы искусственного интеллекта",
        "электронная цифровая подпись",
    ]
    from .coursework import preprocess_text

    query_results = []
    for query in sample_queries:
        query_words = preprocess_text(query)["lemmas"]
        query_results.append(
            {"query": query, "lemmas": query_words, "results": search_by_vector(
                query_words, models["Word2Vec"], idf, search_articles, article_vectors, top_k=5
            )}
        )
    (output_dir / "sample_search_results.json").write_text(
        json.dumps(query_results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "training_articles": len(articles),
        "training_tokens": sum(len(article["lemmas"]) for article in articles),
        "vector_size": models["Word2Vec"].vector_size,
        "models": comparison,
        "retrieval": retrieval,
        "sample_query_count": len(sample_queries),
        "limitations": [
            "Корпус мал для устойчивых эмбеддингов; результаты зависят от seed и состава актов.",
            "Релевантность поиска автоматически оценивается по принадлежности к тому же акту, а не экспертами.",
            "PCA сохраняет лишь часть многомерной структуры и показывает тенденции, а не строгие кластеры.",
        ],
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary
