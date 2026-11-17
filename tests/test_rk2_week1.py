import unittest

import numpy as np

from lawankz.rk2_week1 import evaluate_similarity, select_balanced_articles, split_into_chunks


class Rk2Week1Tests(unittest.TestCase):
    def test_balanced_selection_returns_three_articles_per_document(self):
        articles = []
        document_ids = (
            "personal_data",
            "cybersecurity",
            "communications",
            "artificial_intelligence",
            "access_to_information",
        )
        for document_id in document_ids:
            for number, words in enumerate((60, 80, 110, 160, 240, 300), start=1):
                articles.append(
                    {
                        "document_id": document_id,
                        "article_id": f"{document_id}:{number}",
                        "clean_word_count": words,
                    }
                )
        selected = select_balanced_articles(articles)
        self.assertEqual(len(selected), 15)
        self.assertEqual({item["clean_word_count"] for item in selected}, {80, 160, 240})

    def test_chunking_preserves_all_words(self):
        text = "Первое короткое предложение. Второе предложение немного длиннее. Третье завершает пример."
        chunks = split_into_chunks(text, max_words=5)
        self.assertGreater(len(chunks), 1)
        self.assertEqual(" ".join(chunks).split(), text.split())

    def test_similarity_metrics_reward_clear_groups(self):
        vectors = np.asarray(
            [
                [1.0, 0.0],
                [0.99, 0.01],
                [0.98, 0.02],
                [0.0, 1.0],
                [0.01, 0.99],
                [0.02, 0.98],
            ]
        )
        metrics = evaluate_similarity(vectors, ["a", "a", "a", "b", "b", "b"])
        self.assertEqual(metrics["precision_at_2"], 1.0)
        self.assertEqual(metrics["hit_rate_at_2"], 1.0)
        self.assertGreater(metrics["cosine_separation"], 0.9)


if __name__ == "__main__":
    unittest.main()
