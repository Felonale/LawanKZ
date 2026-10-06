"""Тесты первого прототипа на коротких синтетических фрагментах."""

import unittest

from lawankz.preprocessing import analyze_text, normalize_text, split_articles


class PreprocessingTests(unittest.TestCase):
    def test_normalization_preserves_line_boundaries(self):
        self.assertEqual(normalize_text(" Статья 1.  Тест\n\n  Текст\tакта "), "Статья 1. Тест\nТекст акта")

    def test_empty_input_is_rejected(self):
        with self.assertRaises(ValueError):
            split_articles(" \n\t ")

    def test_article_headings_are_separated(self):
        articles = split_articles(
            "Статья 1. Общие положения\nПервый текст.\nСтатья 2-1. Защита\nВторой текст."
        )
        self.assertEqual([article["number"] for article in articles], ["1", "2-1"])
        self.assertEqual(articles[1]["text"], "Второй текст.")

    def test_plain_fragment_is_kept(self):
        articles = split_articles("Оператор обязан защищать данные.")
        self.assertIsNone(articles[0]["number"])

    def test_requirement_candidates_and_neutral_sentence(self):
        result = analyze_text(
            "Статья 1. Данные\nОператор обязан защищать данные. "
            "Обработка включает хранение данных. Не допускается передача третьим лицам."
        )
        self.assertEqual(result["article_count"], 1)
        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(
            [candidate["markers"][0] for candidate in result["candidate_requirements"]],
            ["обязан", "не допускается"],
        )


if __name__ == "__main__":
    unittest.main()
