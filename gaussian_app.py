"""Логика PyCharm/desktop-интерфейса на PyQt для моделирования гауссовых кадров."""

from dataclasses import dataclass, field

import matplotlib.patches as patches
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
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
    QVBoxLayout,
    QWidget,
)

from gaussian_math import crop_around_max, fit_gaussian_weighted, lsb_to_watts, model_image, normalize_signal_sum1


@dataclass
class GaussianFrameSimulator:
    """Состояние модели кадра с временным и фиксируемым геометрическим шумом."""

    geometric_seed: int = 42
    geometric_noise: np.ndarray | None = None
    frame_index: int = 0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def reset_geometric_noise(self):
        self.geometric_noise = None

    def set_geometric_seed(self, seed):
        """Меняет seed геометрического шума и сбрасывает сохраненную карту шума."""
        if self.geometric_seed != seed:
            self.geometric_seed = seed
            self.reset_geometric_noise()

    def next_frame(self):
        self.frame_index += 1

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
                geom_rng = np.random.default_rng(self.geometric_seed)
                self.geometric_noise = geom_rng.normal(0.0, geometric_noise_lsb, shape)
            geometric = self.geometric_noise
        else:
            geometric = self.rng.normal(0.0, geometric_noise_lsb, shape)
            self.geometric_noise = None

        temporal = self.rng.normal(0.0, temporal_noise_lsb, shape)
        return np.clip(clean + geometric + temporal, 0.0, 2**adc_bits - 1)


class GaussianSimulatorWindow(QMainWindow):
    """Аккуратное PyQt-окно с числовыми полями и тремя графиками."""

    def __init__(self, config):
        super().__init__()
        self.config = dict(config)
        self.simulator = GaussianFrameSimulator(geometric_seed=self.config["GEOMETRIC_SEED"])
        self.last_frame = None
        self.last_roi = None
        self.last_roi_without_background = None
        self.last_fit = None
        self.inputs = {}
        self.setWindowTitle("Модель гауссова кадра")
        self.resize(1500, 900)
        self._build_ui()
        self.update_model(run_fit=True)

    def _build_ui(self):
        root = QWidget(self)
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(10)

        controls = QGroupBox("Параметры модели")
        controls_layout = QGridLayout(controls)
        controls_layout.setHorizontalSpacing(18)
        controls_layout.setVerticalSpacing(8)
        root_layout.addWidget(controls)

        field_specs = [
            ("width", "Ширина, px", "WIDTH", 3, 4096, 1, int),
            ("height", "Высота, px", "HEIGHT", 3, 4096, 1, int),
            ("x0", "X0, px", "X0", 0.0, 4095.0, 0.1, float),
            ("y0", "Y0, px", "Y0", 0.0, 4095.0, 0.1, float),
            ("sigma", "Sigma, px", "SIGMA_PX", 0.05, 200.0, 0.05, float),
            ("amplitude_lsb", "Амплитуда пика, LSB", "AMPLITUDE_LSB", 0.0, self.config["MAX_ADC_CODE"], 1.0, float),
            ("background_lsb", "Фон пикселя, LSB", "BACKGROUND_LSB", 0.0, self.config["MAX_ADC_CODE"], 1.0, float),
            ("temporal_noise_lsb", "Временной шум σ, LSB", "TEMPORAL_NOISE_LSB", 0.0, 100000.0, 1.0, float),
            ("geometric_noise_lsb", "Геометрический шум σ, LSB", "GEOMETRIC_NOISE_LSB", 0.0, 100000.0, 1.0, float),
            ("geometric_seed", "Geometric seed", "GEOMETRIC_SEED", 0, 2**31 - 1, 1, int),
            ("lsb_per_picowatt", "LSB/пВт", "LSB_PER_PICOWATT", 0.000001, 1000000.0, 1.0, float),
        ]

        columns = 4
        for index, spec in enumerate(field_specs):
            key, label, config_key, minimum, maximum, step, value_type = spec
            row = index // columns
            column = (index % columns) * 2
            controls_layout.addWidget(QLabel(label), row, column)
            widget = self._make_spin_box(value_type, minimum, maximum, step, self.config[config_key])
            controls_layout.addWidget(widget, row, column + 1)
            self.inputs[key] = widget
            widget.valueChanged.connect(self._on_value_changed)

        self.fix_geometric_checkbox = QCheckBox("Фиксировать геометрический шум")
        self.fix_geometric_checkbox.setChecked(self.config["FIX_GEOMETRIC_NOISE"])
        self.fix_geometric_checkbox.setToolTip(
            "Если включено, одна и та же карта геометрического шума используется во всех кадрах до сброса."
        )
        self.fix_geometric_checkbox.stateChanged.connect(self._on_fix_geometric_changed)
        controls_layout.addWidget(self.fix_geometric_checkbox, 3, 0, 1, 2)

        button_row = QHBoxLayout()
        self.fit_button = QPushButton("Пересчитать sigma")
        self.new_frame_button = QPushButton("Новый кадр")
        self.reset_geom_button = QPushButton("Сброс геом. шума")
        self.fit_button.clicked.connect(lambda: self.update_model(run_fit=True))
        self.new_frame_button.clicked.connect(self._on_new_frame_clicked)
        self.reset_geom_button.clicked.connect(self._on_reset_geom_clicked)
        button_row.addWidget(self.fit_button)
        button_row.addWidget(self.new_frame_button)
        button_row.addWidget(self.reset_geom_button)
        button_row.addStretch(1)
        root_layout.addLayout(button_row)

        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        root_layout.addWidget(self.info_label)

        self.figure = Figure(figsize=(14, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        root_layout.addWidget(self.canvas, stretch=1)
        self.frame_axis, self.roi_axis, self.fit_axis = self.figure.subplots(1, 3)

        self.setCentralWidget(root)

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
            spin_box.setDecimals(6)
            spin_box.setValue(float(value))
        spin_box.setKeyboardTracking(False)
        spin_box.setMinimumWidth(130)
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
            "geometric_seed": int(self.inputs["geometric_seed"].value()),
            "adc_bits": adc_bits,
            "lsb_per_picowatt": max(float(self.inputs["lsb_per_picowatt"].value()), 1e-12),
        }

    def update_model(self, run_fit=False):
        params = self._params()
        self.simulator.set_geometric_seed(params["geometric_seed"])
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
        for axis in (self.frame_axis, self.roi_axis, self.fit_axis):
            axis.clear()

        self.frame_axis.imshow(self.last_frame, cmap="gray", vmin=0, vmax=2**params["adc_bits"] - 1)
        self.frame_axis.set_title("Зашумленный кадр, LSB")

        self.roi_axis.imshow(self.last_roi_without_background, cmap="gray")
        self.roi_axis.set_title("Обрезка 3×3, фон вычтен")

        self.fit_axis.imshow(self.last_fit["model"], cmap="gray")
        self.fit_axis.set_title(f"Расчет: σ={self.last_fit['sigma']:.3f} px")
        circle = patches.Circle(
            (self.last_fit["x0"], self.last_fit["y0"]),
            self.last_fit["sigma"],
            edgecolor="r",
            facecolor="none",
            linestyle="--",
        )
        self.fit_axis.add_patch(circle)

        for axis in (self.frame_axis, self.roi_axis, self.fit_axis):
            axis.set_xticks([])
            axis.set_yticks([])

        amp_watts = float(lsb_to_watts(params["amplitude_lsb"], params["lsb_per_picowatt"]))
        background_watts = float(lsb_to_watts(params["background_lsb"], params["lsb_per_picowatt"]))
        temporal_watts = float(lsb_to_watts(params["temporal_noise_lsb"], params["lsb_per_picowatt"]))
        self.info_label.setText(
            f"Кадр {self.simulator.frame_index}; максимум x={max_x}, y={max_y}; "
            f"амплитуда пика={amp_watts:.3e} Вт; фон/пиксель={background_watts:.3e} Вт; "
            f"временной шум σ={temporal_watts:.3e} Вт; "
            f"geometric_seed={params['geometric_seed']}; "
            f"геом. шум {'фиксирован' if self.fix_geometric_checkbox.isChecked() else 'меняется'}"
        )
        self.canvas.draw_idle()

    def _on_value_changed(self):
        self.last_fit = None
        self.update_model(run_fit=True)

    def _on_fix_geometric_changed(self):
        if not self.fix_geometric_checkbox.isChecked():
            self.simulator.reset_geometric_noise()
        self.last_fit = None
        self.update_model(run_fit=True)

    def _on_new_frame_clicked(self):
        self.simulator.next_frame()
        self.last_fit = None
        self.update_model(run_fit=True)

    def _on_reset_geom_clicked(self):
        self.simulator.reset_geometric_noise()
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