import unittest
from types import SimpleNamespace

import torch

from lawankz.rk2_week5 import confidence_level, predict_text, validate_input_text


class FakeTokenizer:
    def __call__(self, texts, **kwargs):
        return {"input_ids": torch.ones((len(texts), 3), dtype=torch.long)}


class FakeModel:
    config = SimpleNamespace(id2label={0: "class_a", 1: "class_b"})

    def __call__(self, **kwargs):
        count = kwargs["input_ids"].shape[0]
        logits = torch.tensor([[0.0, 2.0]]).repeat(count, 1)
        return SimpleNamespace(logits=logits)


class Rk2Week5Tests(unittest.TestCase):
    def test_short_input_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_input_text("слишком коротко")

    def test_confidence_levels_are_readable(self):
        self.assertEqual(confidence_level(0.80), "высокая")
        self.assertEqual(confidence_level(0.50), "средняя")
        self.assertEqual(confidence_level(0.30), "низкая")

    def test_prediction_returns_ranked_probabilities(self):
        result = predict_text(
            "Это достаточно длинный контрольный юридический текст для проверки.",
            FakeTokenizer(),
            FakeModel(),
        )
        self.assertEqual(result["predicted_label"], "class_b")
        self.assertGreater(result["probabilities"][0]["probability"], 0.8)
        self.assertAlmostEqual(
            sum(row["probability"] for row in result["probabilities"]), 1.0, places=5
        )


if __name__ == "__main__":
    unittest.main()
