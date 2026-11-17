import unittest
from collections import Counter

import numpy as np

from lawankz.rk2_week4 import (
    build_experiment_specs,
    classify_error_cause,
    stratified_subset_indices,
)


class Rk2Week4Tests(unittest.TestCase):
    def test_stratified_subset_keeps_every_class(self):
        articles = [
            {"document_id": label, "text": "Текст", "clean_word_count": 100}
            for label in ("a", "b", "c")
            for _ in range(10)
        ]
        selected = stratified_subset_indices(articles, np.arange(30), 0.5)
        labels = [articles[int(index)]["document_id"] for index in selected]
        self.assertEqual(Counter(labels), Counter({"a": 5, "b": 5, "c": 5}))

    def test_experiment_plan_reuses_shared_configuration(self):
        specs = build_experiment_specs()
        self.assertEqual(len(specs), 5)
        pairs = {(spec["learning_rate"], spec["data_fraction"]) for spec in specs}
        self.assertEqual(len(pairs), 5)
        self.assertIn((3e-5, 1.0), pairs)

    def test_short_text_has_explainable_error_cause(self):
        article = {
            "document_id": "minority",
            "text": "Общее правило без дополнительных пояснений.",
            "clean_word_count": 6,
        }
        cause, explanation = classify_error_cause(
            article, Counter({"minority": 2, "majority": 10})
        )
        self.assertEqual(cause, "короткий общий контекст")
        self.assertIn("6", explanation)


if __name__ == "__main__":
    unittest.main()
