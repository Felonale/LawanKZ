"""Минимальные проверки конвейера недель 2–4."""

import unittest

import spacy

from lawankz.coursework import prepare_articles, preprocess_text


class CourseworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nlp = spacy.load("ru_core_news_sm", disable=["ner", "parser"])

    def test_legal_negation_is_not_removed_as_stopword(self):
        result = preprocess_text(
            "Оператор не должен передавать данные без согласия субъекта.", self.nlp
        )
        for word in ("не", "должен", "данные", "без"):
            self.assertIn(word, result["words"])
        self.assertEqual(len(result["words"]), len(result["lemmas"]))
        self.assertEqual(len(result["words"]), len(result["stems"]))

    def test_articles_keep_document_reference(self):
        document = {
            "id": "sample",
            "title": "Учебный текст",
            "text": (
                "Статья 1. Общие положения\n"
                "Оператор обязан защищать персональные данные и обеспечивать безопасность "
                "информации при обработке персональных данных граждан Республики Казахстан."
            ),
        }
        articles = prepare_articles([document])
        self.assertEqual(len(articles), 1)
        self.assertEqual(articles[0]["document_id"], "sample")
        self.assertEqual(articles[0]["article_number"], "1")


if __name__ == "__main__":
    unittest.main()
