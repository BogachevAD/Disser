"""Логика PyCharm/desktop-интерфейса для моделирования гауссовых кадров."""

from dataclasses import dataclass, field

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, TextBox

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


class GaussianSimulatorApp:
    """Matplotlib-интерфейс, который можно запускать из PyCharm как обычный .py файл."""

    NUMERIC_FIELDS = [
        ("width", "Ширина, px", "WIDTH", int),
        ("height", "Высота, px", "HEIGHT", int),
        ("x0", "X0, px", "X0", float),
        ("y0", "Y0, px", "Y0", float),
        ("sigma", "Sigma, px", "SIGMA_PX", float),
        ("amplitude_lsb", "Амплитуда пика, LSB", "AMPLITUDE_LSB", float),
        ("background_lsb", "Фон пикселя, LSB", "BACKGROUND_LSB", float),
        ("temporal_noise_lsb", "Временной шум σ, LSB", "TEMPORAL_NOISE_LSB", float),
        ("geometric_noise_lsb", "Геометрический шум σ, LSB", "GEOMETRIC_NOISE_LSB", float),
        ("lsb_per_picowatt", "LSB/пВт", "LSB_PER_PICOWATT", float),
    ]

    def __init__(self, config):
        self.config = dict(config)
        self.values = {
            field_key: caster(self.config[config_key])
            for field_key, _label, config_key, caster in self.NUMERIC_FIELDS
        }
        self.simulator = GaussianFrameSimulator(geometric_seed=self.config["GEOMETRIC_SEED"])
        self.fix_geometric_noise = self.config["FIX_GEOMETRIC_NOISE"]
        self.last_frame = None
        self.last_roi = None
        self.last_roi_without_background = None
        self.last_fit = None
        self.text_boxes = {}
        self._build_layout()
        self.update(run_fit=True)

    def show(self):
        plt.show()

    def _build_layout(self):
        self.figure = plt.figure(figsize=(15, 8), constrained_layout=False)
        self.figure.canvas.manager.set_window_title("Модель гауссова кадра")
        self.figure.suptitle(
            "Параметры: амплитуда — пиковая добавка над фоном; фон — постоянный уровень каждого пикселя",
            fontsize=11,
        )

        self.frame_axis = self.figure.add_axes([0.04, 0.08, 0.28, 0.53])
        self.roi_axis = self.figure.add_axes([0.37, 0.08, 0.25, 0.53])
        self.fit_axis = self.figure.add_axes([0.68, 0.08, 0.25, 0.53])
        self.info_axis = self.figure.add_axes([0.04, 0.63, 0.90, 0.04])
        self.info_axis.axis("off")

        start_x = 0.08
        start_y = 0.91
        cell_w = 0.22
        cell_h = 0.04
        gap_x = 0.08
        gap_y = 0.065
        columns = 3

        for index, (field_key, label, _config_key, _caster) in enumerate(self.NUMERIC_FIELDS):
            row = index // columns
            column = index % columns
            left = start_x + column * (cell_w + gap_x)
            bottom = start_y - row * gap_y
            axis = self.figure.add_axes([left, bottom, cell_w, cell_h])
            text_box = TextBox(axis, label, initial=str(self.values[field_key]))
            text_box.on_submit(self._make_text_submit_handler(field_key))
            self.text_boxes[field_key] = text_box

        self.check_axis = self.figure.add_axes([0.68, 0.69, 0.24, 0.05])
        self.check = CheckButtons(self.check_axis, ["Фиксировать геометрический шум"], [self.fix_geometric_noise])
        self.check.on_clicked(self._on_check_clicked)

        self.fit_button_axis = self.figure.add_axes([0.08, 0.69, 0.18, 0.05])
        self.new_frame_axis = self.figure.add_axes([0.30, 0.69, 0.18, 0.05])
        self.reset_geom_axis = self.figure.add_axes([0.52, 0.69, 0.12, 0.05])
        self.fit_button = Button(self.fit_button_axis, "Пересчитать sigma")
        self.new_frame_button = Button(self.new_frame_axis, "Новый кадр")
        self.reset_geom_button = Button(self.reset_geom_axis, "Сброс геом.")
        self.fit_button.on_clicked(self._on_fit_clicked)
        self.new_frame_button.on_clicked(self._on_new_frame_clicked)
        self.reset_geom_button.on_clicked(self._on_reset_geom_clicked)

    def _make_text_submit_handler(self, field_key):
        def handler(text):
            self._on_text_submitted(field_key, text)

        return handler

    def _on_text_submitted(self, field_key, text):
        caster = self._caster_for(field_key)
        value = caster(float(text)) if caster is int else caster(text)
        self.values[field_key] = value
        self.last_fit = None
        self.update(run_fit=True)

    def _caster_for(self, field_key):
        for key, _label, _config_key, caster in self.NUMERIC_FIELDS:
            if key == field_key:
                return caster
        return float

    def _params(self):
        adc_bits = self.config["ADC_BITS"]
        max_code = 2**adc_bits - 1
        width = max(3, int(self.values["width"]))
        height = max(3, int(self.values["height"]))
        return {
            "width": width,
            "height": height,
            "x0": min(max(float(self.values["x0"]), 0.0), width - 1),
            "y0": min(max(float(self.values["y0"]), 0.0), height - 1),
            "sigma": max(float(self.values["sigma"]), 0.05),
            "amplitude_lsb": min(max(float(self.values["amplitude_lsb"]), 0.0), max_code),
            "background_lsb": min(max(float(self.values["background_lsb"]), 0.0), max_code),
            "temporal_noise_lsb": max(float(self.values["temporal_noise_lsb"]), 0.0),
            "geometric_noise_lsb": max(float(self.values["geometric_noise_lsb"]), 0.0),
            "adc_bits": adc_bits,
            "lsb_per_picowatt": max(float(self.values["lsb_per_picowatt"]), 1e-12),
        }

    def update(self, run_fit=False):
        params = self._params()
        self.last_frame = self.simulator.simulate(
            params["width"], params["height"], params["x0"], params["y0"], params["sigma"],
            params["amplitude_lsb"], params["background_lsb"], params["temporal_noise_lsb"],
            params["geometric_noise_lsb"], self.fix_geometric_noise, params["adc_bits"],
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
        self.frame_axis.plot(params["x0"], params["y0"], "ro", markersize=3)
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
        self.info_axis.clear()
        self.info_axis.axis("off")
        self.info_axis.text(
            0.01,
            0.5,
            f"Кадр {self.simulator.frame_index}; максимум x={max_x}, y={max_y}; "
            f"амплитуда пика={amp_watts:.3e} Вт; фон/пиксель={background_watts:.3e} Вт; "
            f"временной шум σ={temporal_watts:.3e} Вт; "
            f"геом. шум {'фиксирован' if self.fix_geometric_noise else 'меняется'}",
            va="center",
            fontsize=10,
        )
        self.figure.canvas.draw_idle()

    def _on_check_clicked(self, label):
        self.fix_geometric_noise = not self.fix_geometric_noise
        if not self.fix_geometric_noise:
            self.simulator.reset_geometric_noise()
        self.last_fit = None
        self.update(run_fit=True)

    def _on_new_frame_clicked(self, event):
        self.simulator.next_frame()
        self.last_fit = None
        self.update(run_fit=True)

    def _on_reset_geom_clicked(self, event):
        self.simulator.reset_geometric_noise()
        self.last_fit = None
        self.update(run_fit=True)

    def _on_fit_clicked(self, event):
        self.update(run_fit=True)


def run_gaussian_simulator(config):
    """Запускает интерактивное окно matplotlib."""
    app = GaussianSimulatorApp(config)
    app.show()
    return app