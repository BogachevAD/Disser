"""PyQt6-интерфейс и генератор кадров для модели гауссова пятна.

Модуль связывает параметры оптического пятна, шумы фотоприёмной матрицы,
выбор ROI и субпиксельную оценку. Интерфейс показывает истинные, обнаруженные
и восстановленные координаты, чтобы ошибки выбора окна нельзя было принять
за отсутствие расчёта.
"""

from dataclasses import dataclass, field

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

# PyQt6 импортируется раньше QtAgg: так Matplotlib однозначно выбирает уже
# загруженный Qt-биндинг и в обычном Python, и внутри пакета PyInstaller.
import matplotlib.patches as patches
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from gaussian_math import (
    FIT_METHOD_NELDER_MEAD,
    FIT_METHOD_QUADRANT_NELDER_MEAD,
    ROI_MODE_MATCHED_FILTER,
    ROI_MODE_TRUTH,
    estimate_background_ring,
    fit_gaussian,
    local_to_global,
    lsb_to_watts,
    model_image,
    select_roi,
)
from algorithm_animation import AlgorithmAnimationDialog


@dataclass(frozen=True)
class NoiseMapRecord:
    """Описывает воспроизводимую карту шума без хранения большого массива.

    map_id — номер в интерфейсе, seed — состояние генерации, shape — (height, width).
    По этим данным единичная карта N(0,1) восстанавливается побитно одинаково.
    """

    map_id: int
    seed: int
    shape: tuple[int, int]


@dataclass
class GaussianFrameSimulator:
    """Хранит генератор, текущие карты и историю реализаций шумов.

    Карты содержат N(0,1), а их масштаб задаётся в LSB при simulate(). В истории
    хранятся seed и размер последних 10 карт каждого типа, а не тяжёлые массивы.
    """

    geometric_noise: np.ndarray | None = None
    rng: np.random.Generator = field(default_factory=np.random.default_rng)
    temporal_noise: np.ndarray | None = None
    geometric_noise_history: list[NoiseMapRecord] = field(default_factory=list)
    temporal_noise_history: list[NoiseMapRecord] = field(default_factory=list)
    geometric_noise_map_id: int | None = None
    temporal_noise_map_id: int | None = None
    noise_map_counter: int = 0

    def _generate_noise_map(self, shape, noise_kind):
        """Создаёт и кэширует единичную карту заданного noise_kind.

        shape задаёт (height,width), noise_kind равен temporal или geometric;
        возвращается массив N(0,1), а история соответствующего типа ограничена 10.
        """
        if noise_kind not in {"temporal", "geometric"}:
            raise ValueError(f"Неизвестный тип шума: {noise_kind}")
        shape = tuple(int(value) for value in shape)
        seed = int(self.rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64))
        noise_map = np.random.default_rng(seed).standard_normal(shape)
        self.noise_map_counter += 1
        record = NoiseMapRecord(self.noise_map_counter, seed, shape)
        history = getattr(self, f"{noise_kind}_noise_history")
        history.insert(0, record)
        del history[10:]
        setattr(self, f"{noise_kind}_noise", noise_map)
        setattr(self, f"{noise_kind}_noise_map_id", record.map_id)
        return noise_map

    def noise_map_records(self, noise_kind, shape=None):
        """Возвращает историю temporal/geometric, при необходимости по размеру.

        noise_kind выбирает независимый кэш; shape=(height,width) скрывает карты,
        несовместимые с текущим кадром, не удаляя их из десяти последних записей.
        """
        if noise_kind not in {"temporal", "geometric"}:
            raise ValueError(f"Неизвестный тип шума: {noise_kind}")
        records = list(getattr(self, f"{noise_kind}_noise_history"))
        if shape is None:
            return records
        normalized_shape = tuple(int(value) for value in shape)
        return [record for record in records if record.shape == normalized_shape]

    def select_noise_map(self, noise_kind, map_id):
        """Восстанавливает выбранную карту из seed и делает её текущей.

        noise_kind задаёт кэш, map_id приходит из выпадающего списка. Возвращает
        True при найденной записи и False, если карта уже вытеснена из истории.
        """
        for record in self.noise_map_records(noise_kind):
            if record.map_id == map_id:
                noise_map = np.random.default_rng(record.seed).standard_normal(record.shape)
                setattr(self, f"{noise_kind}_noise", noise_map)
                setattr(self, f"{noise_kind}_noise_map_id", record.map_id)
                return True
        return False

    def clear_temporal_noise(self):
        """Снимает текущую фиксацию временного рисунка, сохраняя его в истории.

        Входных переменных нет; следующий нефиксированный расчёт создаст новую карту.
        """
        self.temporal_noise = None
        self.temporal_noise_map_id = None

    def clear_geometric_noise(self):
        """Снимает текущую фиксацию пространственного рисунка шума.

        Входных переменных нет; кэш сохраняется для последующего выбора карты.
        """
        self.geometric_noise = None
        self.geometric_noise_map_id = None

    def generate_temporal_noise(self, shape, temporal_noise_lsb=None):
        """Создаёт и кэширует единичную временную карту N(0,1).

        shape задаёт размер; temporal_noise_lsb оставлен симметрично интерфейсу,
        но масштаб применяется позднее, чтобы менять СКО при том же рисунке.
        """
        return self._generate_noise_map(shape, "temporal")

    def generate_geometric_noise(self, shape, geometric_noise_lsb=None):
        """Создаёт единичную карту N(0,1) размера shape=(height,width).

        geometric_noise_lsb сохранён в сигнатуре для совместимости, но масштаб
        применяется позднее: это позволяет менять sigma, не меняя рисунок карты.
        """
        return self._generate_noise_map(shape, "geometric")

    def simulate(
        self, width, height, x0, y0, sigma, amplitude_lsb, background_lsb,
        temporal_noise_lsb, geometric_noise_lsb, fix_geometric_noise, adc_bits,
        fix_temporal_noise=False,
    ):
        """Формирует один синтетический кадр фотоприёмной матрицы.

        width/height задают размер; x0/y0/sigma — гауссову ФРТ; amplitude_lsb и
        background_lsb — пик и фон; два noise_lsb — СКО шумов; fix_* управляют
        повторным использованием карт; adc_bits задаёт насыщение. Возвращается
        float-матрица после сложения и ограничения АЦП.
        """
        shape = (height, width)

        # Оптический блок: нормированная ФРТ переводится в пиковую амплитуду LSB.
        gaussian = model_image(shape, x0, y0, sigma)
        gaussian_peak = np.max(gaussian)
        if gaussian_peak > 0:
            gaussian = gaussian / gaussian_peak
        clean = background_lsb + amplitude_lsb * gaussian

        # Каждый шум получает новую карту либо использует зафиксированную реализацию.
        if fix_geometric_noise:
            if self.geometric_noise is None or self.geometric_noise.shape != shape:
                self.generate_geometric_noise(shape)
            geometric = geometric_noise_lsb * self.geometric_noise
        else:
            geometric = geometric_noise_lsb * self.generate_geometric_noise(shape)
        if fix_temporal_noise:
            if self.temporal_noise is None or self.temporal_noise.shape != shape:
                self.generate_temporal_noise(shape)
            temporal = temporal_noise_lsb * self.temporal_noise
        else:
            temporal = temporal_noise_lsb * self.generate_temporal_noise(shape)

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
    FIT_METHODS = [
        ("Нелдер–Мид — взвешенный МНК", FIT_METHOD_NELDER_MEAD),
        ("Квадранты → Нелдер–Мид", FIT_METHOD_QUADRANT_NELDER_MEAD),
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
        self.last_background_stats = None
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

        # Общая сетка удерживает каждую численную таблицу под её изображением.
        visual_grid = QGridLayout()
        visual_grid.setHorizontalSpacing(10)
        visual_grid.setVerticalSpacing(6)
        root_layout.addLayout(visual_grid, stretch=1)
        for column in range(4):
            visual_grid.setColumnStretch(column, 1)

        # Четыре оси показывают полный кадр, ROI, оценку и модель в общей шкале LSB.
        self.figure = Figure(figsize=(14, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        visual_grid.addWidget(self.canvas, 0, 0, 1, 4)
        self.frame_axis, self.roi_axis, self.fit_axis, self.model_axis = self.figure.subplots(1, 4)

        # Под первым изображением таблицы нет; матрицы 2–4 занимают те же четверти.
        visual_grid.addWidget(QWidget(), 1, 0)
        roi_group, self.roi_matrix = self._make_matrix_table("Матрица изображения 2, LSB")
        fit_group, self.fit_matrix = self._make_matrix_table("Матрица изображения 3, LSB")
        model_group, self.model_matrix = self._make_matrix_table("Матрица изображения 4, LSB")
        visual_grid.addWidget(roi_group, 1, 1)
        visual_grid.addWidget(fit_group, 1, 2)
        visual_grid.addWidget(model_group, 1, 3)
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
        """Добавляет фиксацию, историю и генерацию карт обоих шумов.

        layout — сетка группы «Шумы»; созданные checkbox/combo/button сохраняются
        в self и управляют текущими картами GaussianFrameSimulator.
        """
        self.fix_temporal_checkbox = QCheckBox("Фиксировать временной шум")
        self.fix_temporal_checkbox.setChecked(self.config.get("FIX_TEMPORAL_NOISE", False))
        self.fix_temporal_checkbox.stateChanged.connect(self._on_fix_temporal_changed)
        layout.addWidget(self.fix_temporal_checkbox, 2, 0, 1, 2)
        layout.addWidget(QLabel("Кэш временного"), 3, 0)
        self.temporal_noise_combo = QComboBox()
        self.temporal_noise_combo.setToolTip("Последние 10 временных карт текущего сеанса и размера кадра")
        self.temporal_noise_combo.currentIndexChanged.connect(self._on_temporal_noise_selected)
        layout.addWidget(self.temporal_noise_combo, 3, 1)
        self.generate_temporal_button = QPushButton("Новая временная карта")
        self.generate_temporal_button.clicked.connect(self._on_generate_temporal_clicked)
        layout.addWidget(self.generate_temporal_button, 4, 0, 1, 2)

        self.fix_geometric_checkbox = QCheckBox("Фиксировать геометрический шум")
        self.fix_geometric_checkbox.setChecked(self.config["FIX_GEOMETRIC_NOISE"])
        self.fix_geometric_checkbox.stateChanged.connect(self._on_fix_geometric_changed)
        layout.addWidget(self.fix_geometric_checkbox, 5, 0, 1, 2)
        layout.addWidget(QLabel("Кэш геометрического"), 6, 0)
        self.geometric_noise_combo = QComboBox()
        self.geometric_noise_combo.setToolTip("Последние 10 геометрических карт текущего сеанса и размера кадра")
        self.geometric_noise_combo.currentIndexChanged.connect(self._on_geometric_noise_selected)
        layout.addWidget(self.geometric_noise_combo, 6, 1)
        self.generate_geom_button = QPushButton("Новая карта геометрического шума")
        self.generate_geom_button.clicked.connect(self._on_generate_geometric_clicked)
        layout.addWidget(self.generate_geom_button, 7, 0, 1, 2)

    def _refresh_noise_cache_controls(self, shape):
        """Обновляет два выпадающих списка для текущего shape=(height,width).

        Карты других размеров остаются в кэше, но не предлагаются, поскольку их
        нельзя без изменения статистики наложить на текущий кадр.
        """
        self._refresh_noise_combo(
            self.temporal_noise_combo, "temporal", "T",
            self.simulator.temporal_noise_map_id, shape,
        )
        self._refresh_noise_combo(
            self.geometric_noise_combo, "geometric", "G",
            self.simulator.geometric_noise_map_id, shape,
        )

    def _refresh_noise_combo(self, combo, noise_kind, prefix, current_id, shape):
        """Заполняет combo совместимыми записями одного кэша.

        noise_kind выбирает историю, prefix формирует краткое имя, current_id
        сохраняет выбранную карту, shape фильтрует несовместимые размеры.
        """
        records = self.simulator.noise_map_records(noise_kind, shape)
        combo.blockSignals(True)
        combo.clear()
        for record in records:
            height, width = record.shape
            combo.addItem(f"{prefix}#{record.map_id} — {width}×{height}", record.map_id)
        if records:
            selected_index = combo.findData(current_id)
            combo.setCurrentIndex(selected_index if selected_index >= 0 else 0)
            combo.setEnabled(True)
        else:
            combo.addItem("Нет совместимых карт", None)
            combo.setEnabled(False)
        combo.blockSignals(False)

    def _add_roi_controls(self, parent_layout):
        """Добавляет режим ROI, метод fit и обработку фоновой рамки.

        parent_layout — верхняя строка, принимающая новую группу виджетов.
        """
        group = QGroupBox("ROI, фон и оценивание")
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

        layout.addWidget(QLabel("Метод fit"), 2, 0)
        self.fit_method_combo = QComboBox()
        for label, value in self.FIT_METHODS:
            self.fit_method_combo.addItem(label, value)
        configured_method = self.config.get("FIT_METHOD", FIT_METHOD_NELDER_MEAD)
        self.fit_method_combo.setCurrentIndex(max(0, self.fit_method_combo.findData(configured_method)))
        self.fit_method_combo.currentIndexChanged.connect(self._on_value_changed)
        layout.addWidget(self.fit_method_combo, 2, 1)

        layout.addWidget(QLabel("Толщина рамки"), 3, 0)
        self.ring_width_combo = QComboBox()
        for width in (1, 2, 3, 4):
            self.ring_width_combo.addItem(f"{width} px", width)
        configured_width = self.config.get("BACKGROUND_RING_WIDTH", 1)
        self.ring_width_combo.setCurrentIndex(max(0, self.ring_width_combo.findData(configured_width)))
        self.ring_width_combo.currentIndexChanged.connect(self._on_value_changed)
        layout.addWidget(self.ring_width_combo, 3, 1)

        layout.addWidget(QLabel("Отступ до рамки"), 4, 0)
        self.ring_gap_combo = QComboBox()
        for gap in (0, 1, 2, 3, 4):
            self.ring_gap_combo.addItem(f"{gap} px", gap)
        configured_gap = self.config.get("BACKGROUND_RING_GAP", 3)
        self.ring_gap_combo.setCurrentIndex(max(0, self.ring_gap_combo.findData(configured_gap)))
        self.ring_gap_combo.currentIndexChanged.connect(self._on_value_changed)
        layout.addWidget(self.ring_gap_combo, 4, 1)

        self.subtract_background_checkbox = QCheckBox("Вычитать средний фон по рамке")
        self.subtract_background_checkbox.setChecked(self.config.get("SUBTRACT_RING_BACKGROUND", True))
        self.subtract_background_checkbox.setToolTip(
            "Вычитается среднее значение рамки; случайная реализация шума в ROI при этом остаётся."
        )
        self.subtract_background_checkbox.stateChanged.connect(self._on_value_changed)
        layout.addWidget(self.subtract_background_checkbox, 5, 0, 1, 2)

        self.use_noise_checkbox = QCheckBox("Учитывать СКО рамки в SNR и χ²")
        self.use_noise_checkbox.setChecked(self.config.get("USE_RING_NOISE", True))
        self.use_noise_checkbox.setToolTip(
            "Одно и то же СКО для всех пикселей не меняет минимум, но задаёт физический масштаб SNR и χ²."
        )
        self.use_noise_checkbox.stateChanged.connect(self._on_value_changed)
        layout.addWidget(self.use_noise_checkbox, 6, 0, 1, 2)

        # Кнопка относится к выбранному fit и поэтому расположена в этой группе.
        self.animation_button = QPushButton("Показать работу алгоритма")
        self.animation_button.setToolTip("Открыть пошаговую визуализацию последнего расчёта")
        self.animation_button.clicked.connect(self._on_animation_clicked)
        layout.addWidget(self.animation_button, 7, 0, 1, 2)

    def _make_matrix_table(self, title):
        """Создаёт группу с read-only таблицей численной матрицы.

        title подписывает связанную картинку; возвращаются QGroupBox и таблица,
        которую _set_matrix_table() заполняет значениями и индексами пикселей.
        """
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        table = QTableWidget()
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setMinimumHeight(150)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(table)
        return group, table

    def _set_matrix_table(self, table, matrix):
        """Заполняет table элементами двумерной matrix в инженерном формате.

        Индексы строк и столбцов соответствуют локальным координатам ROI; каждый
        QTableWidgetItem центрируется и выводит значение LSB с тремя знаками.
        """
        values = np.asarray(matrix, dtype=float)
        rows, columns = values.shape
        table.setRowCount(rows)
        table.setColumnCount(columns)
        table.setHorizontalHeaderLabels([f"x={index}" for index in range(columns)])
        table.setVerticalHeaderLabels([f"y={index}" for index in range(rows)])
        for row in range(rows):
            for column in range(columns):
                item = QTableWidgetItem(f"{values[row, column]:.3f}")
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row, column, item)

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
            "fix_temporal_noise": self.fix_temporal_checkbox.isChecked(),
            "fix_geometric_noise": self.fix_geometric_checkbox.isChecked(),
            "adc_bits": adc_bits,
            "lsb_per_picowatt": max(float(self.inputs["lsb_per_picowatt"].value()), 1e-12),
            "roi_mode": self.roi_mode_combo.currentData(),
            "roi_size": int(self.roi_size_combo.currentData()),
            "fit_method": self.fit_method_combo.currentData(),
            "ring_width": int(self.ring_width_combo.currentData()),
            "ring_gap": int(self.ring_gap_combo.currentData()),
            "subtract_background": self.subtract_background_checkbox.isChecked(),
            "use_ring_noise": self.use_noise_checkbox.isChecked(),
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
            params["geometric_noise_lsb"], params["fix_geometric_noise"], params["adc_bits"],
            params["fix_temporal_noise"],
        )
        self._refresh_noise_cache_controls((params["height"], params["width"]))
        self.last_selection = select_roi(
            self.last_frame, params["roi_mode"], params["roi_size"], params["x0"], params["y0"],
            params["sigma"], params["background_lsb"],
        )
        self.last_roi = self.last_selection.roi
        self.last_background_stats = estimate_background_ring(
            self.last_frame, self.last_selection.center_x, self.last_selection.center_y,
            params["roi_size"], params["ring_width"], params["ring_gap"],
        )
        self.last_fit = fit_gaussian(
            self.last_roi,
            method=params["fit_method"],
            background_level=self.last_background_stats.mean,
            subtract_background=params["subtract_background"],
            noise_sigma=self.last_background_stats.std if params["use_ring_noise"] else None,
        )
        self.last_roi_without_background = self.last_fit["fit_signal"]
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
        roi_half = params["roi_size"] // 2
        ring_inner = roi_half + params["ring_gap"]
        ring_outer = ring_inner + params["ring_width"]
        self.frame_axis.add_patch(
            patches.Rectangle(
                (self.last_selection.origin_x - 0.5, self.last_selection.origin_y - 0.5),
                params["roi_size"], params["roi_size"], edgecolor="yellow",
                facecolor="none", linewidth=1.0,
            )
        )
        for radius, linestyle in ((ring_inner, ":"), (ring_outer, "-")):
            self.frame_axis.add_patch(
                patches.Rectangle(
                    (self.last_selection.center_x - radius - 0.5,
                     self.last_selection.center_y - radius - 0.5),
                    2 * radius + 1, 2 * radius + 1, edgecolor="magenta",
                    facecolor="none", linewidth=0.8, linestyle=linestyle,
                )
            )
        self.frame_axis.set_title("1. Кадр и выбранные области")

        # ROI и модель используют одну шкалу LSB, поэтому их яркости сравнимы напрямую.
        signal_min = float(np.min(self.last_roi_without_background))
        signal_max = float(np.max(self.last_roi_without_background))
        if signal_max <= signal_min:
            signal_max = signal_min + 1.0
        self.roi_axis.imshow(self.last_roi_without_background, cmap="gray", vmin=signal_min, vmax=signal_max)
        self.roi_axis.set_title(
            f"2. Вход fit {params['roi_size']}×{params['roi_size']}, начало "
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
        self._set_matrix_table(self.roi_matrix, self.last_roi_without_background)
        self._set_matrix_table(self.fit_matrix, self.last_roi_without_background)
        self._set_matrix_table(self.model_matrix, self.last_fit["model_signal"])
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
        method_name = self.fit_method_combo.currentText()
        self.position_label.setText(
            f"Расчёт №{self.calculation_index}; ROI: {mode_name}; метод: {method_name}; "
            f"задано ({params['x0']:.3f}, {params['y0']:.3f}); "
            f"сырой max=({raw_x}, {raw_y}); центр ROI=({self.last_selection.center_x}, "
            f"{self.last_selection.center_y}); локальная оценка=({self.last_fit['x0']:.3f}, "
            f"{self.last_fit['y0']:.3f}); глобальная оценка=({global_x:.3f}, {global_y:.3f}); "
            f"Δ=({delta_x:+.3f}, {delta_y:+.3f}) px, |Δ|={center_error:.3f} px; "
            f"fit={'OK' if self.last_fit['success'] else 'ОШИБКА'}, loss={self.last_fit['loss']:.3e}."
        )
        quadrant = self.last_fit.get("quadrant")
        if quadrant is not None:
            selected = "/".join(quadrant["selected_quadrants"])
            self.position_label.setText(
                self.position_label.text()
                + f" Квадранты: {selected}; Δ/Σ=({quadrant['delta_x']:+.3f}, "
                  f"{quadrant['delta_y']:+.3f}); старт=({quadrant['x0_init']:.3f}, "
                  f"{quadrant['y0_init']:.3f}); уверенность={quadrant['confidence']:.3f}."
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
            f"рамка: mean={self.last_background_stats.mean:.3f}, median="
            f"{self.last_background_stats.median:.3f}, σ={self.last_background_stats.std:.3f} LSB, "
            f"N={self.last_background_stats.pixel_count}, отступ={params['ring_gap']} px; фон "
            f"{'вычтен' if params['subtract_background'] else 'не вычтен'}; "
            f"SNRpeak={self.last_fit['snr_peak']:.3f}, χ²red="
            f"{self.last_fit['reduced_chi_square']:.3f}; задано σ={params['sigma']:.3f}, "
            f"оценено σ={self.last_fit['sigma']:.3f} px."
        )

    def _on_value_changed(self, *_):
        """Обрабатывает изменение любого параметра.

        *_ принимает необязательное значение Qt-сигнала; запускается полный расчёт.
        """
        self.update_model()

    def _on_fix_temporal_changed(self, *_):
        """Обрабатывает флажок фиксации временного шума.

        *_ содержит состояние Qt; снятие флажка сбрасывает текущий выбор, после
        чего пересчёт создаёт и кэширует новую независимую временную реализацию.
        """
        if not self.fix_temporal_checkbox.isChecked():
            self.simulator.clear_temporal_noise()
        self.update_model()

    def _on_generate_temporal_clicked(self):
        """Создаёт новую временную карту и сразу фиксирует её.

        Размер берётся из интерфейса; карта попадает в начало кэша, после чего
        выполняется один расчёт с этой же, а не следующей реализацией.
        """
        params = self._params()
        self.simulator.generate_temporal_noise((params["height"], params["width"]))
        self.fix_temporal_checkbox.blockSignals(True)
        self.fix_temporal_checkbox.setChecked(True)
        self.fix_temporal_checkbox.blockSignals(False)
        self.update_model()

    def _on_temporal_noise_selected(self, *_):
        """Восстанавливает временную карту, выбранную пользователем в combo.

        *_ принимает индекс Qt; найденная map_id становится текущей, а фиксация
        включается без промежуточного расчёта с посторонней реализацией.
        """
        map_id = self.temporal_noise_combo.currentData()
        if map_id is None or not self.simulator.select_noise_map("temporal", map_id):
            return
        self.fix_temporal_checkbox.blockSignals(True)
        self.fix_temporal_checkbox.setChecked(True)
        self.fix_temporal_checkbox.blockSignals(False)
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

    def _on_geometric_noise_selected(self, *_):
        """Восстанавливает геометрическую карту, выбранную пользователем.

        *_ принимает индекс Qt; map_id читается из combo, карта становится текущей,
        и её фиксация включается перед единственным последующим пересчётом.
        """
        map_id = self.geometric_noise_combo.currentData()
        if map_id is None or not self.simulator.select_noise_map("geometric", map_id):
            return
        self.fix_geometric_checkbox.blockSignals(True)
        self.fix_geometric_checkbox.setChecked(True)
        self.fix_geometric_checkbox.blockSignals(False)
        self.update_model()

    def _on_animation_clicked(self):
        """Открывает анимацию последнего рассчитанного метода.

        last_fit содержит неизменяемые вход, квадранты и трассу оптимизации;
        модальный диалог не генерирует новый кадр и не запускает повторный fit.
        """
        if self.last_fit is None:
            return
        dialog = AlgorithmAnimationDialog(self.last_fit, self)
        dialog.exec()


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
