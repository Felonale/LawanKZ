import unittest

import numpy as np

from lawankz.rk2_week3 import (
    build_chunk_records,
    class_weights_from_records,
    split_train_validation,
)


class Rk2Week3Tests(unittest.TestCase):
    def test_validation_is_created_only_from_train_pool(self):
        articles = [
            {"document_id": label, "article_id": f"{label}:{index}"}
            for label in ("a", "b", "c")
            for index in range(15)
        ]
        train_pool = np.arange(len(articles))
        fit, validation = split_train_validation(articles, train_pool)
        self.assertFalse(set(fit) & set(validation))
        self.assertEqual(set(fit) | set(validation), set(train_pool))
        self.assertEqual(
            {articles[int(index)]["document_id"] for index in validation},
            {"a", "b", "c"},
        )

    def test_chunks_keep_article_and_label(self):
        articles = [
            {"document_id": "law", "text": "Первое предложение. Второе предложение."}
        ]
        records = build_chunk_records(articles, np.asarray([0]), {"law": 4})
        self.assertTrue(records)
        self.assertTrue(all(record["article_index"] == 0 for record in records))
        self.assertTrue(all(record["label"] == 4 for record in records))

    def test_class_weights_increase_for_smaller_class(self):
        records = [
            {"label": 0},
            {"label": 0},
            {"label": 0},
            {"label": 1},
        ]
        weights = class_weights_from_records(records, 2).numpy()
        self.assertGreater(weights[1], weights[0])


if __name__ == "__main__":
    unittest.main()
