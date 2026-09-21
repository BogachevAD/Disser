"""Проверяет единый источник README для встроенной справки приложения."""

from pathlib import Path
import unittest

from method_help import load_method_help_markdown, markdown_for_qt


class MethodHelpTests(unittest.TestCase):
    """Проверяет извлечение раздела и адаптацию формул для QTextBrowser."""

    def test_project_readme_contains_extractable_help_section(self):
        """Извлекает реальный раздел без внешнего сворачиваемого HTML-контейнера."""
        readme_path = Path(__file__).resolve().parents[1] / "README.md"
        markdown = load_method_help_markdown(readme_path)
        self.assertIn("### Главное различие", markdown)
        self.assertIn("### 24. Прямое сравнение реализаций", markdown)
        self.assertNotIn("METHOD_HELP_START", markdown)
        self.assertNotIn("<details>", markdown)

    def test_latex_is_preserved_as_readable_qt_code(self):
        """Преобразует блочную и встроенную формулы без потери их содержания."""
        prepared = markdown_for_qt("До $x_0$ после.\n\n$$\nsigma=exp(eta)\n$$")
        self.assertIn("`x_0`", prepared)
        self.assertIn("```text\nsigma=exp(eta)\n```", prepared)


if __name__ == "__main__":
    unittest.main()
