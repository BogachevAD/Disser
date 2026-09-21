"""Проверяет единый источник README для встроенной справки приложения."""

from pathlib import Path
import unittest

from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QTextDocument

from method_help import (
    load_method_help_markdown,
    markdown_for_qt,
    normalize_mathtext_formula,
    render_formula_image,
)


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

    def test_latex_has_readable_fallback_without_document(self):
        """Сохраняет текст формулы, если QTextDocument не был передан."""
        prepared = markdown_for_qt("До $x_0$ после.\n\n$$\nsigma=exp(eta)\n$$")
        self.assertIn("`x_0`", prepared)
        self.assertIn("```text\nsigma=exp(eta)\n```", prepared)

    def test_formula_is_registered_as_qt_image(self):
        """Заменяет LaTeX ссылкой на реальный графический ресурс документа."""
        document = QTextDocument()
        prepared = markdown_for_qt(
            r"$$\varepsilon_x=\frac{Q_R-Q_L}{Q_R+Q_L}$$", document
        )
        self.assertIn("formula://method-help/0", prepared)
        self.assertIn("formula://method-help/0", document.toHtml())
        image = document.resource(
            QTextDocument.ResourceType.ImageResource,
            QUrl("formula://method-help/0"),
        )
        self.assertFalse(image.isNull())

    def test_all_embedded_readme_formulas_render(self):
        """Рендерит каждую формулу справки, включая cases и матрицы."""
        readme_path = Path(__file__).resolve().parents[1] / "README.md"
        markdown = load_method_help_markdown(readme_path)
        document = QTextDocument()
        prepared = markdown_for_qt(markdown, document)
        self.assertNotIn("```text", prepared)
        self.assertGreaterEqual(prepared.count("formula://method-help/"), 140)

    def test_mathtext_normalizes_unsupported_latex_constructs(self):
        """Преобразует cases, bmatrix и boxed без потери основных символов."""
        source = (
            r"\boxed{w(r)=\begin{cases}1,& |r|\leq\delta,\\"
            r"\dfrac{\delta}{|r|},& |r|>\delta.\end{cases}}"
        )
        normalized = normalize_mathtext_formula(source)
        self.assertNotIn(r"\begin", normalized)
        self.assertNotIn(r"\boxed", normalized)
        self.assertIn(r"\substack", normalized)
        self.assertFalse(render_formula_image(source).isNull())


if __name__ == "__main__":
    unittest.main()
