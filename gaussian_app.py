"""PyQt6-интерфейс и генератор кадров для модели гауссова пятна.

Модуль связывает параметры оптического пятна, шумы фотоприёмной матрицы,
выбор ROI и субпиксельную оценку. Интерфейс показывает истинные, обнаруженные
и восстановленные координаты, чтобы ошибки выбора окна нельзя было принять
за отсутствие расчёта.
"""

from dataclasses import dataclass, field

import matplotlib.patches as patches
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gaussian_math import (
    ROI_MODE_MATCHED_FILTER,
    ROI_MODE_TRUTH,
    fit_gaussian_weighted,
    local_to_global,
    lsb_to_watts,
    model_image,
    select_roi,
)


@dataclass
class GaussianFrameSimulator:
    """Хранит генератор случайных чисел и фиксируемую карту неоднородности.

    geometric_noise содержит одну реализацию N(0,1); rng формирует независимый
    временной шум и новые карты. Реальный масштаб задаётся в LSB при simulate().
    """

    geometric_noise: np.ndarray | None = None
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def clear_geometric_noise(self):
        """Удаляет сохранённый пространственный рисунок шума.

        Входных переменных нет; изменяется поле self.geometric_noise.
        """
        self.geometric_noise = None

    def generate_geometric_noise(self, shape, geometric_noise_lsb=None):
        """Создаёт единичную карту N(0,1) размера shape=(height,width).

        geometric_noise_lsb сохранён в сигнатуре для совместимости, но масштаб
        применяется позднее: это позволяет менять sigma, не меняя рисунок карты.
        """
        self.geometric_noise = self.rng.standard_normal(shape)

    def simulate(
        self, width, height, x0, y0, sigma, amplitude_lsb, background_lsb,
        temporal_noise_lsb, geometric_noise_lsb, fix_geometric_noise, adc_bits,
    ):
        """Формирует один синтетический кадр фотоприёмной матрицы.

        width/height задают размер; x0/y0/sigma — гауссову ФРТ; amplitude_lsb и
        background_lsb — пик и фон; два noise_lsb — СКО шумов; adc_bits задаёт
        насыщение. Возвращается float-матрица после сложения и ограничения АЦП.
        """
        shape = (height, width)

        # Оптический блок: нормированная ФРТ переводится в пиковую амплитуду LSB.
        gaussian = model_image(shape, x0, y0, sigma)
        gaussian_peak = np.max(gaussian)
        if gaussian_peak > 0:
            gaussian = gaussian / gaussian_peak
        clean = background_lsb + amplitude_lsb * gaussian

        # Геометрический шум постоянен по кадрам при фиксации, временной — независим.
        if fix_geometric_noise:
            if self.geometric_noise is None or self.geometric_noise.shape != shape:
                self.generate_geometric_noise(shape)
            geometric = geometric_noise_lsb * self.geometric_noise
        else:
            geometric = self.rng.normal(0.0, geometric_noise_lsb, shape)
            self.geometric_noise = None
        temporal = self.rng.normal(0.0, temporal_noise_lsb, shape)

        # АЦП отсекает отрицательные значения и насыщает сигнал максимальным кодом.
        return np.clip(clean + geometric + temporal, 0.0, 2**adc_bits - 1)


class GaussianSimulatorWindow(QMainWindow):
    """Главное окно настройки, расчёта и диагностики модели.

    config передаёт стартовые константы. Окно пересчитывает кадр при изменении
    поля и хранит последние frame, ROI, fit и глобальную привязку результата.
    """

    # Описание числовых полей: внутреннее имя, подпись, ключ config, min/max, шаг, тип.
    FRAME_FIELDS = [
        ("width", "Ширина, px", "WIDTH", 3, 4096, 1, int),
        ("height", "Высота, px", "HEIGHT", 3, 4096, 1, int),
        ("lsb_per_picowatt", "LSB/пВт", "LSB_PER_PICOWATT", 0.000001, 1000000.0, 1.0, float),
    ]
    GAUSSIAN_FIELDS = [
        ("x0", "X0, px", "X0", 0.0, 4095.0, 0.1, float),
        ("y0", "Y0, px", "Y0", 0.0, 4095.0, 0.1, float),
        ("sigma", "Sigma, px", "SIGMA_PX", 0.05, 200.0, 0.05, float),
        ("amplitude_lsb", "Амплитуда, LSB", "AMPLITUDE_LSB", 0.0, "MAX_ADC_CODE", 1.0, float),
        ("background_lsb", "Фон, LSB", "BACKGROUND_LSB", 0.0, "MAX_ADC_CODE", 1.0, float),
    ]
    NOISE_FIELDS = [
        ("temporal_noise_lsb", "Временной σ, LSB", "TEMPORAL_NOISE_LSB", 0.0, 100000.0, 1.0, float),
        ("geometric_noise_lsb", "Геометрический σ, LSB", "GEOMETRIC_NOISE_LSB", 0.0, 100000.0, 1.0, float),
    ]
    ROI_MODES = [
        ("По заданному центру — верификация", ROI_MODE_TRUTH),
        ("Согласованный фильтр — обнаружение", ROI_MODE_MATCHED_FILTER),
    ]

    def __init__(self, config):
        """Создаёт состояние модели, интерфейс и первый кадр.

        config — словарь стартовых физических и вычислительных параметров.
        """
        super().__init__()
        self.config = dict(config)
        self.simulator = GaussianFrameSimulator()
        self.last_frame = None
        self.last_roi = None
        self.last_roi_without_background = None
        self.last_selection = None
        self.last_fit = None
        self.last_global_fit = None
        self.calculation_index = 0
        self.inputs = {}
        self.setWindowTitle("Модель гауссова кадра и субпиксельной оценки")
        self.resize(1580, 980)
        self._build_ui()
        self.update_model()

    def _build_ui(self):
        """Собирает панели параметров, графики и таблицы.

        Метод не принимает аргументов и записывает созданные виджеты в self.
        """
        root = QWidget(self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(8)

        # Верхняя строка объединяет параметры кадра, ФРТ, шумов и алгоритма ROI.
        controls_row = QHBoxLayout()
        controls_row.setSpacing(10)
        root_layout.addLayout(controls_row)
        self._add_group(controls_row, "Кадр и пересчёт", self.FRAME_FIELDS)
        self._add_group(controls_row, "Гауссоида", self.GAUSSIAN_FIELDS)
        noise_group = self._add_group(controls_row, "Шумы", self.NOISE_FIELDS)
        self._add_noise_controls(noise_group.layout())
        self._add_roi_controls(controls_row)

        # Две строки диагностики разделяют геометрию оценки и радиометрию/шумы.
        self.position_label = QLabel()
        self.position_label.setWordWrap(True)
        self.radiometry_label = QLabel()
        self.radiometry_label.setWordWrap(True)
        root_layout.addWidget(self.position_label)
        root_layout.addWidget(self.radiometry_label)

        # Четыре оси показывают полный кадр, ROI, оценку и модель в общей шкале LSB.
        self.figure = Figure(figsize=(14, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        root_layout.addWidget(self.canvas, stretch=1)
        self.frame_axis, self.roi_axis, self.fit_axis, self.model_axis = self.figure.subplots(1, 4)

        # Нижние моноширинные поля позволяют численно сравнить три матрицы.
        matrix_row = QHBoxLayout()
        matrix_row.setSpacing(12)
        root_layout.addLayout(matrix_row)
        matrix_row.addStretch(1)
        self.roi_matrix = self._make_matrix_box("Матрица изображения 2")
        self.fit_matrix = self._make_matrix_box("Матрица изображения 3")
        self.model_matrix = self._make_matrix_box("Матрица изображения 4")
        matrix_row.addWidget(self.roi_matrix)
        matrix_row.addWidget(self.fit_matrix)
        matrix_row.addWidget(self.model_matrix)
        matrix_row.addStretch(1)
        self.setCentralWidget(root)

    def _add_group(self, parent_layout, title, fields):
        """Создаёт группу spinbox по декларации fields и связывает пересчёт.

        parent_layout принимает группу, title отображается пользователю, а каждый
        элемент fields описывает переменную и допустимые инженерные значения.
        """
        group = QGroupBox(title)
        layout = QGridLayout(group)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(5)
        parent_layout.addWidget(group)
        for row, spec in enumerate(fields):
            key, label, config_key, minimum, maximum, step, value_type = spec
            if maximum == "MAX_ADC_CODE":
                maximum = self.config["MAX_ADC_CODE"]
            layout.addWidget(QLabel(label), row, 0)
            widget = self._make_spin_box(value_type, minimum, maximum, step, self.config[config_key])
            layout.addWidget(widget, row, 1)
            self.inputs[key] = widget
            widget.valueChanged.connect(self._on_value_changed)
        return group

    def _add_noise_controls(self, layout):
        """Добавляет управление геометрическим шумом.

        layout — сетка группы «Шумы», куда помещаются флажок и кнопка.
        """
        self.fix_geometric_checkbox = QCheckBox("Фиксировать геометрический шум")
        self.fix_geometric_checkbox.setChecked(self.config["FIX_GEOMETRIC_NOISE"])
        self.fix_geometric_checkbox.stateChanged.connect(self._on_fix_geometric_changed)
        layout.addWidget(self.fix_geometric_checkbox, 2, 0, 1, 2)
        self.generate_geom_button = QPushButton("Новая карта геометрического шума")
        self.generate_geom_button.clicked.connect(self._on_generate_geometric_clicked)
        layout.addWidget(self.generate_geom_button, 3, 0, 1, 2)

    def _add_roi_controls(self, parent_layout):
        """Добавляет режим ROI и размер окна оценки.

        parent_layout — верхняя строка, принимающая новую группу виджетов.
        """
        group = QGroupBox("ROI и обнаружение")
        layout = QGridLayout(group)
        parent_layout.addWidget(group)
        layout.addWidget(QLabel("Центр окна"), 0, 0)
        self.roi_mode_combo = QComboBox()
        for label, value in self.ROI_MODES:
            self.roi_mode_combo.addItem(label, value)
        configured_mode = self.config.get("ROI_MODE", ROI_MODE_TRUTH)
        self.roi_mode_combo.setCurrentIndex(max(0, self.roi_mode_combo.findData(configured_mode)))
        self.roi_mode_combo.currentIndexChanged.connect(self._on_value_changed)
        layout.addWidget(self.roi_mode_combo, 0, 1)
        layout.addWidget(QLabel("Размер ROI"), 1, 0)
        self.roi_size_combo = QComboBox()
        for size in (3, 5, 7):
            self.roi_size_combo.addItem(f"{size}×{size}", size)
        configured_size = self.config.get("ROI_SIZE", 3)
        self.roi_size_combo.setCurrentIndex(max(0, self.roi_size_combo.findData(configured_size)))
        self.roi_size_combo.currentIndexChanged.connect(self._on_value_changed)
        layout.addWidget(self.roi_size_combo, 1, 1)

    def _make_matrix_box(self, title):
        """Создаёт read-only поле численной матрицы.

        title — начальная подпись; возвращается настроенный QTextEdit.
        """
        box = QTextEdit()
        box.setReadOnly(True)
        box.setFixedHeight(125)
        box.setMinimumWidth(350)
        box.setFont(QFont("Courier New", 9))
        box.setText(title)
        return box

    def _make_spin_box(self, value_type, minimum, maximum, step, value):
        """Создаёт целый или вещественный spinbox.

        Тип, пределы, шаг и начальное value полностью задают редактор числа.
        """
        if value_type is int:
            spin_box = QSpinBox()
            spin_box.setRange(int(minimum), int(maximum))
            spin_box.setSingleStep(int(step))
            spin_box.setValue(int(value))
        else:
            spin_box = QDoubleSpinBox()
            spin_box.setRange(float(minimum), float(maximum))
            spin_box.setSingleStep(float(step))
            spin_box.setDecimals(3)
            spin_box.setValue(float(value))
        spin_box.setKeyboardTracking(False)
        spin_box.setMinimumWidth(88)
        spin_box.setMaximumWidth(110)
        return spin_box

    def _params(self):
        """Считывает и проверяет все поля интерфейса.

        Входных аргументов нет; возвращается словарь параметров текущего кадра.
        """
        adc_bits = self.config["ADC_BITS"]
        max_code = 2**adc_bits - 1
        width = max(3, int(self.inputs["width"].value()))
        height = max(3, int(self.inputs["height"].value()))
        return {
            "width": width,
            "height": height,
            "x0": np.clip(float(self.inputs["x0"].value()), 0.0, width - 1),
            "y0": np.clip(float(self.inputs["y0"].value()), 0.0, height - 1),
            "sigma": max(float(self.inputs["sigma"].value()), 0.05),
            "amplitude_lsb": np.clip(float(self.inputs["amplitude_lsb"].value()), 0.0, max_code),
            "background_lsb": np.clip(float(self.inputs["background_lsb"].value()), 0.0, max_code),
            "temporal_noise_lsb": max(float(self.inputs["temporal_noise_lsb"].value()), 0.0),
            "geometric_noise_lsb": max(float(self.inputs["geometric_noise_lsb"].value()), 0.0),
            "adc_bits": adc_bits,
            "lsb_per_picowatt": max(float(self.inputs["lsb_per_picowatt"].value()), 1e-12),
            "roi_mode": self.roi_mode_combo.currentData(),
            "roi_size": int(self.roi_size_combo.currentData()),
        }

    def update_model(self):
        """Последовательно генерирует кадр, выбирает ROI и оценивает ФРТ.

        В режиме truth окно привязано к заданному центру; в matched_filter оно
        определяется только из кадра. Локальная оценка затем переводится обратно
        в глобальные координаты и передаётся визуализации.
        """
        self.calculation_index += 1
        params = self._params()
        self.last_frame = self.simulator.simulate(
            params["width"], params["height"], params["x0"], params["y0"], params["sigma"],
            params["amplitude_lsb"], params["background_lsb"], params["temporal_noise_lsb"],
            params["geometric_noise_lsb"], self.fix_geometric_checkbox.isChecked(), params["adc_bits"],
        )
        self.last_selection = select_roi(
            self.last_frame, params["roi_mode"], params["roi_size"], params["x0"], params["y0"],
            params["sigma"], params["background_lsb"],
        )
        self.last_roi = self.last_selection.roi
        self.last_roi_without_background = np.clip(
            self.last_roi - params["background_lsb"], 0.0, None
        )
        self.last_fit = fit_gaussian_weighted(self.last_roi_without_background)
        self.last_global_fit = local_to_global(
            self.last_fit["x0"], self.last_fit["y0"],
            self.last_selection.origin_x, self.last_selection.origin_y,
        )
        self._draw(params)

    def _draw(self, params):
        """Перерисовывает изображения, подписи и матрицы.

        params — проверенный словарь текущего запуска из _params().
        """
        for axis in (self.frame_axis, self.roi_axis, self.fit_axis, self.model_axis):
            axis.clear()

        # Полный кадр показывает истинный центр (+) и центр выбранного ROI (квадрат).
        self.frame_axis.imshow(self.last_frame, cmap="gray", vmin=0, vmax=2**params["adc_bits"] - 1)
        self.frame_axis.plot(params["x0"], params["y0"], marker="+", color="cyan", markersize=9, mew=1.5)
        self.frame_axis.plot(
            self.last_selection.center_x, self.last_selection.center_y,
            marker="s", markerfacecolor="none", markeredgecolor="yellow", markersize=9,
        )
        self.frame_axis.set_title("1. Кадр: + задано, □ центр ROI")

        # ROI и модель используют одну шкалу LSB, поэтому их яркости сравнимы напрямую.
        signal_min = float(np.min(self.last_roi_without_background))
        signal_max = float(np.max(self.last_roi_without_background))
        if signal_max <= signal_min:
            signal_max = signal_min + 1.0
        self.roi_axis.imshow(self.last_roi_without_background, cmap="gray", vmin=signal_min, vmax=signal_max)
        self.roi_axis.set_title(
            f"2. ROI {params['roi_size']}×{params['roi_size']}, начало "
            f"({self.last_selection.origin_x}, {self.last_selection.origin_y})"
        )
        self.fit_axis.imshow(self.last_roi_without_background, cmap="gray", vmin=signal_min, vmax=signal_max)
        self.fit_axis.set_title(f"3. Оценка: σ={self.last_fit['sigma']:.3f} px")
        self._draw_fit_overlay(params["roi_size"])
        self.model_axis.imshow(
            self.last_fit["model_signal"], cmap="gray", vmin=signal_min, vmax=signal_max
        )
        self.model_axis.set_title("4. Восстановленная модель, LSB")

        for axis in (self.frame_axis, self.roi_axis, self.fit_axis, self.model_axis):
            axis.set_xticks([])
            axis.set_yticks([])
        self._update_info(params)
        matrix_text = self._format_matrix(self.last_roi_without_background)
        self.roi_matrix.setText("Матрица изображения 2, LSB\n" + matrix_text)
        self.fit_matrix.setText("Матрица изображения 3, LSB\n" + matrix_text)
        self.model_matrix.setText(
            "Матрица изображения 4, LSB\n" + self._format_matrix(self.last_fit["model_signal"])
        )
        self.canvas.draw_idle()

    def _draw_fit_overlay(self, roi_size):
        """Наносит локальный центр, окружность sigma и сетку 15×15.

        roi_size задаёт центральный пиксель окна; координаты x0/y0 и sigma
        берутся из last_fit. Сетка иллюстрирует субпиксели, но не квантует оценку.
        """
        x0, y0, sigma = self.last_fit["x0"], self.last_fit["y0"], self.last_fit["sigma"]
        self.fit_axis.plot(x0, y0, marker="x", color="red", markersize=7, mew=1.5)
        self.fit_axis.add_patch(
            patches.Circle((x0, y0), sigma, edgecolor="red", facecolor="none", linestyle="--", linewidth=1.2)
        )
        center = roi_size // 2
        grid_start = center - 0.5
        for index in range(16):
            coordinate = grid_start + index / 15.0
            self.fit_axis.axvline(
                coordinate, ymin=center / roi_size, ymax=(center + 1) / roi_size,
                color="cyan", linewidth=0.35, alpha=0.75,
            )
            self.fit_axis.axhline(
                coordinate, xmin=center / roi_size, xmax=(center + 1) / roi_size,
                color="cyan", linewidth=0.35, alpha=0.75,
            )

    def _update_info(self, params):
        """Выводит ошибку центра, качество fit и мощности.

        params содержит истинные значения, сравниваемые с last_fit.
        """
        raw_y, raw_x = np.unravel_index(np.argmax(self.last_frame), self.last_frame.shape)
        global_x, global_y = self.last_global_fit
        delta_x, delta_y = global_x - params["x0"], global_y - params["y0"]
        center_error = float(np.hypot(delta_x, delta_y))
        mode_name = self.roi_mode_combo.currentText()
        self.position_label.setText(
            f"Расчёт №{self.calculation_index}; режим: {mode_name}; "
            f"задано ({params['x0']:.3f}, {params['y0']:.3f}); "
            f"сырой max=({raw_x}, {raw_y}); центр ROI=({self.last_selection.center_x}, "
            f"{self.last_selection.center_y}); локальная оценка=({self.last_fit['x0']:.3f}, "
            f"{self.last_fit['y0']:.3f}); глобальная оценка=({global_x:.3f}, {global_y:.3f}); "
            f"Δ=({delta_x:+.3f}, {delta_y:+.3f}) px, |Δ|={center_error:.3f} px; "
            f"fit={'OK' if self.last_fit['success'] else 'ОШИБКА'}, loss={self.last_fit['loss']:.3e}."
        )
        warning = (not self.last_fit["success"]) or center_error > 0.75
        self.position_label.setStyleSheet("color: #b00020;" if warning else "color: #146c2e;")

        amp_watts = float(lsb_to_watts(params["amplitude_lsb"], params["lsb_per_picowatt"]))
        background_watts = float(lsb_to_watts(params["background_lsb"], params["lsb_per_picowatt"]))
        temporal_watts = float(lsb_to_watts(params["temporal_noise_lsb"], params["lsb_per_picowatt"]))
        geometric_watts = float(lsb_to_watts(params["geometric_noise_lsb"], params["lsb_per_picowatt"]))
        self.radiometry_label.setText(
            f"Амплитуда={params['amplitude_lsb']:.3f} LSB ({amp_watts:.3e} Вт); "
            f"фон={params['background_lsb']:.3f} LSB ({background_watts:.3e} Вт); "
            f"временной σ={params['temporal_noise_lsb']:.3f} LSB ({temporal_watts:.3e} Вт); "
            f"геометрический σ={params['geometric_noise_lsb']:.3f} LSB ({geometric_watts:.3e} Вт); "
            f"задано σ={params['sigma']:.3f}, оценено σ={self.last_fit['sigma']:.3f} px."
        )

    def _format_matrix(self, matrix):
        """Форматирует двумерную matrix для QTextEdit.

        Каждый элемент выводится с тремя знаками и фиксированной шириной.
        """
        return "\n".join("  ".join(f"{value:10.3f}" for value in row) for row in matrix)

    def _on_value_changed(self, *_):
        """Обрабатывает изменение любого параметра.

        *_ принимает необязательное значение Qt-сигнала; запускается полный расчёт.
        """
        self.update_model()

    def _on_fix_geometric_changed(self, *_):
        """Обрабатывает флажок фиксации геометрического шума.

        *_ содержит состояние Qt; при снятии флажка карта удаляется.
        """
        if not self.fix_geometric_checkbox.isChecked():
            self.simulator.clear_geometric_noise()
        self.update_model()

    def _on_generate_geometric_clicked(self):
        """Создаёт новую карту неоднородности.

        Входных аргументов нет; фиксация включается и выполняется один пересчёт.
        """
        params = self._params()
        self.simulator.generate_geometric_noise((params["height"], params["width"]))
        self.fix_geometric_checkbox.blockSignals(True)
        self.fix_geometric_checkbox.setChecked(True)
        self.fix_geometric_checkbox.blockSignals(False)
        self.update_model()


GaussianSimulatorApp = GaussianSimulatorWindow


def run_gaussian_simulator(config):
    """Создаёт QApplication, показывает окно с config и запускает event loop.

    config — словарь из run_gauss_simulator.py. Функция возвращает окно после
    завершения приложения, что удобно для интеграционных тестов.
    """
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    window = GaussianSimulatorWindow(config)
    window.show()
    app.exec()
    return window
