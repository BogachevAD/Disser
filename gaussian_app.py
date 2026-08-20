"""Логика PyCharm/desktop-интерфейса на PyQt для моделирования гауссовых кадров."""

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

from gaussian_math import crop_around_max, fit_gaussian_weighted, lsb_to_watts, model_image, normalize_signal_sum1


@dataclass
class GaussianFrameSimulator:
    """Состояние модели кадра с временным и фиксируемым геометрическим шумом."""

    geometric_noise: np.ndarray | None = None
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def clear_geometric_noise(self):
        self.geometric_noise = None

    def generate_geometric_noise(self, shape, geometric_noise_lsb):
        """Создает новую карту геометрического шума для текущего размера кадра."""
        self.geometric_noise = self.rng.normal(0.0, geometric_noise_lsb, shape)

    def simulate(self, width, height, x0, y0, sigma, amplitude_lsb, background_lsb,
                 temporal_noise_lsb, geometric_noise_lsb, fix_geometric_noise, adc_bits):
        """Создает кадр.

        amplitude_lsb — пиковая добавка гауссоиды над фоном в LSB.
        background_lsb — постоянный уровень фона каждого пикселя в LSB.
        """
        shape = (height, width)
        gaussian = model_image(shape, x0, y0, sigma)
        gaussian_peak = np.max(gaussian)
        if gaussian_peak > 0:
            gaussian = gaussian / gaussian_peak
        clean = background_lsb + amplitude_lsb * gaussian

        if fix_geometric_noise:
            if self.geometric_noise is None or self.geometric_noise.shape != shape:
                self.generate_geometric_noise(shape, geometric_noise_lsb)
            geometric = self.geometric_noise
        else:
            geometric = self.rng.normal(0.0, geometric_noise_lsb, shape)
            self.geometric_noise = None

        temporal = self.rng.normal(0.0, temporal_noise_lsb, shape)
        return np.clip(clean + geometric + temporal, 0.0, 2**adc_bits - 1)


class GaussianSimulatorWindow(QMainWindow):
    """Аккуратное PyQt-окно с числовыми полями и тремя графиками."""

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

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.simulator = GaussianFrameSimulator()
        self.last_frame = None
        self.last_roi = None
        self.last_roi_without_background = None
        self.last_fit = None
        self.inputs = {}
        self.setWindowTitle("Модель гауссова кадра")
        self.resize(1500, 950)
        self._build_ui()
        self.update_model(run_fit=True)

    def _build_ui(self):
        root = QWidget(self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(8)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(10)
        root_layout.addLayout(controls_row)
        self._add_group(controls_row, "Кадр и пересчет", self.FRAME_FIELDS)
        self._add_group(controls_row, "Гауссоида", self.GAUSSIAN_FIELDS)
        noise_group = self._add_group(controls_row, "Шумы", self.NOISE_FIELDS)

        noise_layout = noise_group.layout()
        self.fix_geometric_checkbox = QCheckBox("Фиксировать текущий геометрический шум")
        self.fix_geometric_checkbox.setChecked(self.config["FIX_GEOMETRIC_NOISE"])
        self.fix_geometric_checkbox.setToolTip(
            "Если включено, текущая карта геометрического шума остается одной и той же между пересчетами."
        )
        self.fix_geometric_checkbox.stateChanged.connect(self._on_fix_geometric_changed)
        noise_layout.addWidget(self.fix_geometric_checkbox, 2, 0, 1, 2)

        self.generate_geom_button = QPushButton("Сгенерировать геометрический шум")
        self.generate_geom_button.clicked.connect(self._on_generate_geometric_clicked)
        noise_layout.addWidget(self.generate_geom_button, 3, 0, 1, 2)

        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        root_layout.addWidget(self.info_label)

        self.figure = Figure(figsize=(14, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        root_layout.addWidget(self.canvas, stretch=1)
        self.frame_axis, self.roi_axis, self.fit_axis, self.model_axis = self.figure.subplots(1, 4)

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

    def _make_matrix_box(self, title):
        box = QTextEdit()
        box.setReadOnly(True)
        box.setFixedHeight(95)
        box.setMinimumWidth(330)
        box.setFont(QFont("Courier New", 10))
        box.setText(title)
        return box

    def _make_spin_box(self, value_type, minimum, maximum, step, value):
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
        spin_box.setMaximumWidth(105)
        return spin_box

    def _params(self):
        adc_bits = self.config["ADC_BITS"]
        max_code = 2**adc_bits - 1
        width = max(3, int(self.inputs["width"].value()))
        height = max(3, int(self.inputs["height"].value()))
        return {
            "width": width,
            "height": height,
            "x0": min(max(float(self.inputs["x0"].value()), 0.0), width - 1),
            "y0": min(max(float(self.inputs["y0"].value()), 0.0), height - 1),
            "sigma": max(float(self.inputs["sigma"].value()), 0.05),
            "amplitude_lsb": min(max(float(self.inputs["amplitude_lsb"].value()), 0.0), max_code),
            "background_lsb": min(max(float(self.inputs["background_lsb"].value()), 0.0), max_code),
            "temporal_noise_lsb": max(float(self.inputs["temporal_noise_lsb"].value()), 0.0),
            "geometric_noise_lsb": max(float(self.inputs["geometric_noise_lsb"].value()), 0.0),
            "adc_bits": adc_bits,
            "lsb_per_picowatt": max(float(self.inputs["lsb_per_picowatt"].value()), 1e-12),
        }

    def update_model(self, run_fit=False):
        params = self._params()
        self.last_frame = self.simulator.simulate(
            params["width"], params["height"], params["x0"], params["y0"], params["sigma"],
            params["amplitude_lsb"], params["background_lsb"], params["temporal_noise_lsb"],
            params["geometric_noise_lsb"], self.fix_geometric_checkbox.isChecked(), params["adc_bits"],
        )
        self.last_roi, max_x, max_y = crop_around_max(self.last_frame, 3)
        self.last_roi_without_background = np.clip(self.last_roi - params["background_lsb"], 0.0, None)
        if run_fit or self.last_fit is None:
            self.last_fit = fit_gaussian_weighted(normalize_signal_sum1(self.last_roi_without_background))
        self._draw(params, max_x, max_y)

    def _draw(self, params, max_x, max_y):
        for axis in (self.frame_axis, self.roi_axis, self.fit_axis, self.model_axis):
            axis.clear()

        self.frame_axis.imshow(self.last_frame, cmap="gray", vmin=0, vmax=2**params["adc_bits"] - 1)
        self.frame_axis.set_title("1. Зашумленный кадр, LSB")

        vmin = float(np.min(self.last_roi_without_background))
        vmax = float(np.max(self.last_roi_without_background))
        if vmax <= vmin:
            vmax = vmin + 1.0
        self.roi_axis.imshow(self.last_roi_without_background, cmap="gray", vmin=vmin, vmax=vmax)
        self.roi_axis.set_title("2. Обрезка 3×3, фон вычтен")

        self.fit_axis.imshow(self.last_roi_without_background, cmap="gray", vmin=vmin, vmax=vmax)
        self.fit_axis.set_title(f"3. Та же обрезка + σ={self.last_fit['sigma']:.3f} px")
        self._draw_fit_overlay()

        model = self.last_fit["model"]
        model_vmin = float(np.min(model))
        model_vmax = float(np.max(model))
        if model_vmax <= model_vmin:
            model_vmax = model_vmin + 1.0
        self.model_axis.imshow(model, cmap="gray", vmin=model_vmin, vmax=model_vmax)
        self.model_axis.set_title("4. Рассчитанная модель")

        for axis in (self.frame_axis, self.roi_axis, self.fit_axis, self.model_axis):
            axis.set_xticks([])
            axis.set_yticks([])

        self._update_info(params, max_x, max_y)
        matrix_text = self._format_matrix(self.last_roi_without_background)
        self.roi_matrix.setText("Матрица изображения 2\n" + matrix_text)
        self.fit_matrix.setText("Матрица изображения 3\n" + matrix_text)
        self.model_matrix.setText("Матрица изображения 4\n" + self._format_matrix(model))
        self.canvas.draw_idle()

    def _draw_fit_overlay(self):
        x0 = self.last_fit["x0"]
        y0 = self.last_fit["y0"]
        sigma = self.last_fit["sigma"]
        self.fit_axis.plot(x0, y0, marker="x", color="red", markersize=7, mew=1.5)
        circle = patches.Circle((x0, y0), sigma, edgecolor="red", facecolor="none", linestyle="--", linewidth=1.2)
        self.fit_axis.add_patch(circle)
        grid_step = 1.0 / 15.0
        for index in range(16):
            coord = 0.5 + index * grid_step
            self.fit_axis.axvline(coord, ymin=1 / 3, ymax=2 / 3, color="cyan", linewidth=0.35, alpha=0.75)
            self.fit_axis.axhline(coord, xmin=1 / 3, xmax=2 / 3, color="cyan", linewidth=0.35, alpha=0.75)

    def _update_info(self, params, max_x, max_y):
        amp_watts = float(lsb_to_watts(params["amplitude_lsb"], params["lsb_per_picowatt"]))
        background_watts = float(lsb_to_watts(params["background_lsb"], params["lsb_per_picowatt"]))
        temporal_watts = float(lsb_to_watts(params["temporal_noise_lsb"], params["lsb_per_picowatt"]))
        geometric_watts = float(lsb_to_watts(params["geometric_noise_lsb"], params["lsb_per_picowatt"]))
        self.info_label.setText(
            f"max: x={max_x}, y={max_y}; "
            f"амплитуда={params['amplitude_lsb']:.3f} LSB ({amp_watts:.3e} Вт); "
            f"фон={params['background_lsb']:.3f} LSB ({background_watts:.3e} Вт); "
            f"временной σ={params['temporal_noise_lsb']:.3f} LSB ({temporal_watts:.3e} Вт); "
            f"геометрический σ={params['geometric_noise_lsb']:.3f} LSB ({geometric_watts:.3e} Вт); "
            f"геом. шум {'зафиксирован' if self.fix_geometric_checkbox.isChecked() else 'перегенерируется'}"
        )

    def _format_matrix(self, matrix):
        return "\n".join("  ".join(f"{value:10.3f}" for value in row) for row in matrix)

    def _on_value_changed(self):
        self.last_fit = None
        self.update_model(run_fit=True)

    def _on_fix_geometric_changed(self):
        if not self.fix_geometric_checkbox.isChecked():
            self.simulator.clear_geometric_noise()
        self.last_fit = None
        self.update_model(run_fit=True)

    def _on_generate_geometric_clicked(self):
        params = self._params()
        self.simulator.generate_geometric_noise((params["height"], params["width"]), params["geometric_noise_lsb"])
        self.fix_geometric_checkbox.setChecked(True)
        self.last_fit = None
        self.update_model(run_fit=True)


GaussianSimulatorApp = GaussianSimulatorWindow


def run_gaussian_simulator(config):
    """Запускает интерактивное PyQt-окно."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    window = GaussianSimulatorWindow(config)
    window.show()
    app.exec()
    return window