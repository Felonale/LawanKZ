"""Тесты простого Markdown-отчёта."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from lawankz.markdown import render_markdown


class MarkdownTests(unittest.TestCase):
    def test_renders_articles_and_candidates(self):
        result = {
            "source_file": "sample.txt",
            "articles": [
                {"number": "1", "title": "Защита данных", "text": "Оператор обязан защищать данные."},
                {"number": "2", "title": "Термины", "text": "Данные хранятся в системе."},
            ],
            "candidate_requirements": [
                {"article_number": "1", "sentence": "Оператор обязан защищать данные.", "markers": ["обязан"]}
            ],
            "warning": "Требуется проверка человеком.",
        }

        report = render_markdown(result)

        self.assertIn("Источник: sample.txt", report)
        self.assertIn("Статей: 2", report)
        self.assertIn("## Статья 1. Защита данных", report)
        self.assertIn("- Оператор обязан защищать данные. (маркер: обязан)", report)
        self.assertIn("## Статья 2. Термины", report)
        self.assertIn("Явных маркеров не найдено.", report)
        self.assertIn("Примечание: Требуется проверка человеком.", report)

    def test_renders_fragment_without_article_number(self):
        report = render_markdown({
            "articles": [{"number": None, "title": None, "text": "Оператор должен ответить."}],
            "candidate_requirements": [
                {"article_number": None, "sentence": "Оператор должен ответить.", "markers": ["должен"]}
            ],
        })
        self.assertIn("## Фрагмент без номера", report)
        self.assertIn("маркер: должен", report)

    def test_rejects_unrelated_json(self):
        with self.assertRaises(ValueError):
            render_markdown({"other": "data"})

    def test_saved_json_converts_to_same_markdown_as_direct_output(self):
        project_root = Path(__file__).resolve().parents[1]
        sample = project_root / "examples" / "synthetic_act.txt"

        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = Path(temp_dir) / "analysis.json"
            converted_path = Path(temp_dir) / "converted.md"
            direct_path = Path(temp_dir) / "direct.md"

            commands = [
                [sys.executable, "-m", "lawankz", str(sample), "--output", str(json_path)],
                [sys.executable, "-m", "lawankz.markdown", str(json_path), "--output", str(converted_path)],
                [sys.executable, "-m", "lawankz", str(sample), "--output", str(direct_path)],
            ]
            for command in commands:
                completed = subprocess.run(command, cwd=project_root, capture_output=True, text=True)
                self.assertEqual(completed.returncode, 0, completed.stderr)

            self.assertEqual(
                converted_path.read_text(encoding="utf-8"),
                direct_path.read_text(encoding="utf-8"),
            )
            self.assertEqual(
                direct_path.read_text(encoding="utf-8"),
                (project_root / "examples" / "synthetic_act.report.md").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
