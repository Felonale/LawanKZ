"""Проверки извлечения текста без сетевых запросов."""

import unittest

from lawankz.corpus import extract_act_text


class CorpusTests(unittest.TestCase):
    def test_extracts_body_without_navigation_and_notes(self):
        html = """
        <nav>Поиск закона и меню сайта</nav>
        <div class="text_upd"><article>
          <p>Статья 1. Данные</p>
          <p>{body}</p>
          <p class="note">Сноска. Редакционное примечание.</p>
          <p>Данные <a href="/other">подлежат</a> защите.</p>
        </article></div>
        """.format(body="Оператор обязан защищать персональные данные. " * 210)
        text = extract_act_text(html)
        self.assertIn("Статья 1. Данные", text)
        self.assertIn("Данные подлежат защите.", text)
        self.assertNotIn("меню сайта", text)
        self.assertNotIn("Редакционное примечание", text)

    def test_rejects_pages_without_current_act_container(self):
        with self.assertRaisesRegex(ValueError, "Не найдено тело акта"):
            extract_act_text("<div class='text_yts'><article>Утративший силу</article></div>")


if __name__ == "__main__":
    unittest.main()
