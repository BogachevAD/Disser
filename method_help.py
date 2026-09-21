"""Окно встроенной справки по классическому и робастному оцениванию ФРТ.

Текст берётся из отмеченного блока README, поэтому документация приложения и
репозитория имеет один источник. При сборке PyInstaller README упаковывается в
EXE и извлекается во временный каталог вместе с остальными ресурсами.
"""

from pathlib import Path
import re
import sys

from PyQt6.QtGui import QTextCursor
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


def markdown_for_qt(markdown):
    """Адаптирует GitHub Markdown к возможностям QTextDocument.

    Qt отображает заголовки, списки и таблицы, но не вычисляет LaTeX. Блочные
    формулы превращаются в отдельные моноширинные панели, а короткие формулы —
    во встроенный код: обозначения сохраняются полностью и остаются читаемыми.
    """
    prepared = re.sub(
        r"\$\$\s*(.*?)\s*\$\$",
        lambda match: "\n```text\n" + match.group(1).strip() + "\n```\n",
        markdown,
        flags=re.DOTALL,
    )
    prepared = re.sub(
        r"(?<!\$)\$([^$\n]+)\$(?!\$)",
        lambda match: "`" + match.group(1).strip() + "`",
        prepared,
    )
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
            "Справка загружается из README. Формулы показаны в моноширинных "
            "математических блоках без потери обозначений."
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
        self.browser.setMarkdown(markdown_for_qt(source))
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
