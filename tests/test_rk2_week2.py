import unittest

import numpy as np

from lawankz.rk2_week2 import (
    aggregate_chunk_vectors,
    calculate_metrics,
    split_dataset,
    split_text,
)


class Rk2Week2Tests(unittest.TestCase):
    def test_stratified_split_has_no_overlap_and_keeps_classes(self):
        articles = [
            {"document_id": label, "article_id": f"{label}:{index}"}
            for label in ("a", "b", "c")
            for index in range(12)
        ]
        train, test = split_dataset(articles)
        self.assertFalse(set(train) & set(test))
        self.assertEqual(len(train) + len(test), len(articles))
        test_labels = {articles[int(index)]["document_id"] for index in test}
        self.assertEqual(test_labels, {"a", "b", "c"})

    def test_long_sentence_is_split_without_losing_words(self):
        text = " ".join(f"слово{index}" for index in range(23))
        chunks = split_text(text, max_words=10)
        self.assertEqual([len(chunk.split()) for chunk in chunks], [10, 10, 3])
        self.assertEqual(" ".join(chunks), text)

    def test_chunk_vectors_are_averaged_and_normalized(self):
        chunks = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 2.0]], dtype=np.float32)
        result = aggregate_chunk_vectors(chunks, [0, 0, 1], 2)
        np.testing.assert_allclose(result, [[1.0, 0.0], [0.0, 1.0]])

    def test_metrics_are_perfect_for_exact_predictions(self):
        metrics = calculate_metrics(["a", "b", "c"], ["a", "b", "c"])
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["f1_macro"], 1.0)


if __name__ == "__main__":
    unittest.main()
