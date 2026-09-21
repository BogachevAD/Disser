"""Окно встроенной справки по классическому и робастному оцениванию ФРТ.

Текст берётся из отмеченного блока README, поэтому документация приложения и
репозитория имеет один источник. При сборке PyInstaller README упаковывается в
EXE и извлекается во временный каталог вместе с остальными ресурсами.
"""

from io import BytesIO
from pathlib import Path
import re
import sys

from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
from PyQt6.QtCore import QUrl
from PyQt6.QtGui import QColor, QImage, QPainter, QPen, QTextCursor, QTextDocument
from PyQt6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextBrowser,
    QVBoxLayout,
)


HELP_START_MARKER = "<!-- METHOD_HELP_START -->"
HELP_END_MARKER = "<!-- METHOD_HELP_END -->"


def application_resource_path(relative_path):
    """Возвращает путь ресурса в исходниках или распакованном PyInstaller EXE.

    relative_path задаётся относительно корня проекта. В обычном Python корнем
    служит каталог модуля, а в one-file сборке — временный каталог sys._MEIPASS.
    """
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return bundle_root / relative_path


def load_method_help_markdown(readme_path=None):
    """Извлекает подробный раздел между служебными маркерами README.

    readme_path позволяет тестам передать временный файл; без него используется
    упакованный README. Возвращается Markdown без внешнего тега details.
    """
    source_path = Path(readme_path) if readme_path is not None else application_resource_path("README.md")
    try:
        readme = source_path.read_text(encoding="utf-8")
    except OSError as error:
        return (
            "# Справка недоступна\n\n"
            f"Не удалось прочитать `{source_path}`.\n\n"
            f"Техническая причина: `{error}`"
        )
    if HELP_START_MARKER not in readme or HELP_END_MARKER not in readme:
        return (
            "# Справка недоступна\n\n"
            "В README отсутствуют маркеры подробного сравнения методов."
        )
    section = readme.split(HELP_START_MARKER, 1)[1].split(HELP_END_MARKER, 1)[0]
    return section.strip()


def _replace_math_environment(formula, environment, left, right):
    """Заменяет cases/bmatrix на поддерживаемую MathText конструкцию.

    formula содержит нормализованный LaTeX; environment задаёт имя окружения,
    left/right — визуальные скобки. Строки собираются через `substack`, а
    разделитель столбцов `&` заменяется математическим интервалом.
    """
    pattern = rf"\\begin\{{{environment}\}}(.*?)\\end\{{{environment}\}}"

    def replacement(match):
        rows = re.split(r"\\\\(?:\[[^]]+\])?", match.group(1))
        rows = [row.strip().replace("&", r"\quad") for row in rows if row.strip()]
        return left + r"\substack{" + r" \\ ".join(rows) + "}" + right

    return re.sub(pattern, replacement, formula)


def normalize_mathtext_formula(formula):
    """Приводит используемый в README LaTeX к диалекту Matplotlib MathText.

    formula — содержимое `$...$` или `$$...$$`. Переносы схлопываются, краткие
    команды получают явные фигурные скобки, а cases/bmatrix преобразуются без
    изменения математического смысла. Результат пригоден для офлайн-отрисовки.
    """
    normalized = " ".join(formula.strip().split())
    normalized = normalized.replace(r"\dfrac", r"\frac")
    normalized = normalized.replace(r"\frac12", r"\frac{1}{2}")
    normalized = re.sub(r"\\overline\s+([A-Za-z])", r"\\overline{\1}", normalized)
    normalized = re.sub(
        r"\\boldsymbol\s*(\\[A-Za-z]+)", r"\\mathbf{\1}", normalized
    )
    normalized = re.sub(r"\\mathbf\s+([A-Za-z])", r"\\mathbf{\1}", normalized)
    normalized = _replace_math_environment(
        normalized, "bmatrix", r"\left[", r"\right]"
    )
    normalized = _replace_math_environment(
        normalized, "cases", r"\left\{", r"\right."
    )

    # MathText не реализует \boxed. Рамка добавляется к изображению стилем Qt,
    # поэтому здесь снимается только самая внешняя команда, если она есть.
    if normalized.startswith(r"\boxed{"):
        prefix = r"\boxed{"
        depth = 1
        closing_index = None
        for index in range(len(prefix), len(normalized)):
            character = normalized[index]
            escaped = index > 0 and normalized[index - 1] == "\\"
            if character == "{" and not escaped:
                depth += 1
            elif character == "}" and not escaped:
                depth -= 1
                if depth == 0:
                    closing_index = index
                    break
        if closing_index is not None:
            normalized = normalized[len(prefix):closing_index] + normalized[closing_index + 1:]
    return normalized


def render_formula_image(formula, device_pixel_ratio=2.0):
    """Рендерит одну LaTeX-формулу в прозрачный QImage высокого разрешения.

    formula передаётся без внешних долларов; device_pixel_ratio задаёт
    масштаб Retina/HiDPI. Matplotlib MathText работает без LaTeX и интернета,
    поэтому изображение доступно и в автономном EXE.
    """
    boxed = formula.strip().startswith(r"\boxed{")
    normalized = normalize_mathtext_formula(formula)
    output = BytesIO()
    math_to_image(
        f"${normalized}$",
        output,
        format="png",
        dpi=200,
        prop=FontProperties(size=14),
        color="#111111",
    )
    image = QImage.fromData(output.getvalue(), "PNG")
    if image.isNull():
        raise ValueError("Matplotlib вернул пустое изображение формулы")
    if boxed:
        padding = 10
        framed = QImage(
            image.width() + 2 * padding,
            image.height() + 2 * padding,
            QImage.Format.Format_ARGB32,
        )
        framed.fill(QColor(0, 0, 0, 0))
        painter = QPainter(framed)
        painter.drawImage(padding, padding, image)
        painter.setPen(QPen(QColor("#333333"), 2))
        painter.drawRect(1, 1, framed.width() - 3, framed.height() - 3)
        painter.end()
        image = framed
    image.setDevicePixelRatio(device_pixel_ratio)
    return image


def markdown_for_qt(markdown, document=None):
    """Заменяет LaTeX в Markdown ссылками на отрисованные формулы.

    markdown — исходная справка, document — QTextDocument для регистрации
    QImage-ресурсов и установки готовой разметки. Без document функция оставляет
    понятный текстовый fallback. Поддерживаются GitHub-блоки ```math, старые
    блоки $$...$$ и встроенные $...$; неотрисованная формула остаётся кодом.
    """
    formula_index = 0
    image_resources = []

    def image_reference(formula, block):
        nonlocal formula_index
        source = formula.strip()
        if document is None:
            return ("\n```text\n" + source + "\n```\n") if block else f"`{source}`"
        resource_url = QUrl(f"formula://method-help/{formula_index}")
        formula_index += 1
        try:
            image = render_formula_image(source)
        except (ValueError, RuntimeError):
            return ("\n```text\n" + source + "\n```\n") if block else f"`{source}`"
        image_resources.append((resource_url, image))
        alt = "Математическая формула"
        reference = f"![{alt}]({resource_url.toString()})"
        return f"\n\n{reference}\n\n" if block else reference

    # Fenced math — официальный и наиболее устойчивый формат GitHub внутри
    # сворачиваемого HTML-блока <details>. Обрабатываем его раньше обычных fences.
    prepared = re.sub(
        r"```math\s*\n(.*?)\n```",
        lambda match: image_reference(match.group(1), True),
        markdown,
        flags=re.DOTALL,
    )
    prepared = re.sub(
        r"\$\$\s*(.*?)\s*\$\$",
        lambda match: image_reference(match.group(1), True),
        prepared,
        flags=re.DOTALL,
    )
    prepared = re.sub(
        r"(?<!\$)\$([^$\n]+)\$(?!\$)",
        lambda match: image_reference(match.group(1), False),
        prepared,
    )
    if document is not None:
        # setMarkdown очищает ранее добавленные ресурсы, поэтому изображения
        # регистрируются после разбора текста и затем документ перерисовывается.
        document.setMarkdown(prepared)
        for resource_url, image in image_resources:
            document.addResource(
                QTextDocument.ResourceType.ImageResource, resource_url, image
            )
        document.markContentsDirty(0, document.characterCount())
    return prepared


class MethodHelpDialog(QDialog):
    """Показывает прокручиваемое сравнение методов и поиск по его тексту.

    parent связывает окно с главным интерфейсом; markdown при необходимости
    позволяет показать другой подготовленный текст, не обращаясь к README.
    """

    def __init__(self, parent=None, markdown=None):
        super().__init__(parent)
        self.setWindowTitle("Справка: классический и робастный Нелдер–Мид")
        self.resize(1120, 820)
        self.setMinimumSize(760, 560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        heading = QLabel("Классический и робастный Нелдер–Мид: полный разбор")
        heading.setStyleSheet("font-size: 18px; font-weight: 600;")
        layout.addWidget(heading)

        note = QLabel(
            "Справка загружается из README. Формулы автоматически отрисованы "
            "в математической нотации и доступны без интернета."
        )
        note.setWordWrap(True)
        note.setStyleSheet("color: #555;")
        layout.addWidget(note)

        search_row = QHBoxLayout()
        search_row.addWidget(QLabel("Поиск:"))
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Например: функция Хьюбера, фон, IRLS, sigma")
        self.search_input.setClearButtonEnabled(True)
        search_row.addWidget(self.search_input, stretch=1)
        self.search_button = QPushButton("Найти далее")
        search_row.addWidget(self.search_button)
        layout.addLayout(search_row)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setReadOnly(True)
        self.browser.document().setDefaultStyleSheet(
            "body { font-family: 'Segoe UI'; font-size: 10.5pt; }"
            "h3 { color: #17365d; margin-top: 18px; }"
            "pre { font-family: 'Cambria Math', 'Consolas'; background: #f4f6f8; "
            "      border: 1px solid #d8dee4; padding: 8px; }"
            "code { font-family: 'Cambria Math', 'Consolas'; background: #f4f6f8; }"
            "th { background: #eaf0f6; font-weight: 600; }"
            "td, th { padding: 4px; }"
        )
        source = load_method_help_markdown() if markdown is None else markdown
        markdown_for_qt(source, self.browser.document())
        layout.addWidget(self.browser, stretch=1)

        bottom_row = QHBoxLayout()
        self.to_start_button = QPushButton("В начало")
        bottom_row.addWidget(self.to_start_button)
        bottom_row.addStretch(1)
        close_button = QPushButton("Закрыть")
        close_button.setDefault(True)
        bottom_row.addWidget(close_button)
        layout.addLayout(bottom_row)

        self.search_button.clicked.connect(self._find_next)
        self.search_input.returnPressed.connect(self._find_next)
        self.to_start_button.clicked.connect(self._go_to_start)
        close_button.clicked.connect(self.close)
        self._go_to_start()

    def _find_next(self):
        """Ищет следующее вхождение строки из search_input в справочном тексте.

        При достижении конца поиск один раз продолжается с начала документа;
        пустая строка ничего не изменяет.
        """
        query = self.search_input.text().strip()
        if not query:
            return
        if not self.browser.find(query):
            cursor = self.browser.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.Start)
            self.browser.setTextCursor(cursor)
            self.browser.find(query)

    def _go_to_start(self):
        """Перемещает курсор и вертикальную прокрутку к началу справки."""
        cursor = self.browser.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Start)
        self.browser.setTextCursor(cursor)
        self.browser.ensureCursorVisible()
