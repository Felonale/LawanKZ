"""Проверки векторизации и семантического поиска без переобучения моделей."""

import unittest

import numpy as np
from gensim.models import KeyedVectors

from lawankz.embeddings import search_by_vector, weighted_mean_vector


class EmbeddingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = KeyedVectors(vector_size=2)
        cls.model.add_vectors(
            ["данные", "согласие", "связь"],
            np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32),
        )
        cls.idf = {"данные": 2.0, "согласие": 1.0, "связь": 1.0}

    def test_weighted_vector_is_normalized(self):
        vector = weighted_mean_vector(["данные", "согласие"], self.model, self.idf)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)

    def test_unknown_query_returns_none(self):
        self.assertIsNone(weighted_mean_vector(["несуществующее"], self.model, self.idf))

    def test_search_ranks_closest_article_first(self):
        articles = [
            {"article_id": "a", "document_id": "d", "document_title": "Данные", "article_number": "1", "article_title": "", "text": "данные"},
            {"article_id": "b", "document_id": "s", "document_title": "Связь", "article_number": "2", "article_title": "", "text": "связь"},
        ]
        vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        results = search_by_vector(["данные"], self.model, self.idf, articles, vectors, top_k=2)
        self.assertEqual(results[0]["article_id"], "a")


if __name__ == "__main__":
    unittest.main()
