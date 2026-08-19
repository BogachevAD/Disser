"""Логика PyCharm/desktop-интерфейса для моделирования гауссовых кадров."""

from dataclasses import dataclass, field

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, Slider, TextBox

from gaussian_math import crop_around_max, fit_gaussian_weighted, lsb_to_watts, model_image, normalize_pixels_sum1


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
        shape = (height, width)
        clean = background_lsb + amplitude_lsb * model_image(shape, x0, y0, sigma)

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

    def __init__(self, config):
        self.config = dict(config)
        self.simulator = GaussianFrameSimulator(geometric_seed=self.config["GEOMETRIC_SEED"])
        self.fix_geometric_noise = self.config["FIX_GEOMETRIC_NOISE"]
        self.last_frame = None
        self.last_roi = None
        self.last_fit = None
        self._build_layout()
        self.update(run_fit=True)

    def show(self):
        plt.show()

    def _build_layout(self):
        self.figure = plt.figure(figsize=(15, 8))
        self.figure.canvas.manager.set_window_title("Модель гауссова кадра")
        self.frame_axis = self.figure.add_axes([0.04, 0.38, 0.28, 0.55])
        self.roi_axis = self.figure.add_axes([0.37, 0.38, 0.25, 0.55])
        self.fit_axis = self.figure.add_axes([0.68, 0.38, 0.25, 0.55])
        self.info_axis = self.figure.add_axes([0.04, 0.29, 0.89, 0.05])
        self.info_axis.axis("off")

        slider_specs = [
            ("width", "Ширина", self.config["WIDTH"], 3, 256, 1),
            ("height", "Высота", self.config["HEIGHT"], 3, 256, 1),
            ("x0", "X0", self.config["X0"], 0, 255, 0.1),
            ("y0", "Y0", self.config["Y0"], 0, 255, 0.1),
            ("sigma", "Sigma, px", self.config["SIGMA_PX"], 0.1, 20, 0.05),
            ("amplitude_lsb", "Ампл., LSB", self.config["AMPLITUDE_LSB"], 0, self.config["MAX_ADC_CODE"], 10),
            ("background_lsb", "Фон, LSB", self.config["BACKGROUND_LSB"], 0, self.config["MAX_ADC_CODE"], 10),
            ("temporal_noise_lsb", "Врем. шум", self.config["TEMPORAL_NOISE_LSB"], 0, 5000, 1),
            ("geometric_noise_lsb", "Геом. шум", self.config["GEOMETRIC_NOISE_LSB"], 0, 5000, 1),
        ]
        self.sliders = {}
        for idx, (key, label, value, minimum, maximum, step) in enumerate(slider_specs):
            axis = self.figure.add_axes([0.18, 0.23 - idx * 0.023, 0.45, 0.015])
            slider = Slider(axis, label, minimum, maximum, valinit=value, valstep=step)
            slider.on_changed(self._on_slider_changed)
            self.sliders[key] = slider

        self.lsb_axis = self.figure.add_axes([0.76, 0.21, 0.16, 0.04])
        self.lsb_text = TextBox(self.lsb_axis, "LSB/пВт", initial=str(self.config["LSB_PER_PICOWATT"]))
        self.lsb_text.on_submit(self._on_lsb_changed)

        self.check_axis = self.figure.add_axes([0.76, 0.14, 0.18, 0.05])
        self.check = CheckButtons(self.check_axis, ["Фикс. геом. шум"], [self.fix_geometric_noise])
        self.check.on_clicked(self._on_check_clicked)

        self.new_frame_axis = self.figure.add_axes([0.68, 0.06, 0.11, 0.05])
        self.reset_geom_axis = self.figure.add_axes([0.81, 0.06, 0.13, 0.05])
        self.fit_button_axis = self.figure.add_axes([0.52, 0.06, 0.13, 0.05])
        self.new_frame_button = Button(self.new_frame_axis, "Новый кадр")
        self.reset_geom_button = Button(self.reset_geom_axis, "Сброс геом.")
        self.fit_button = Button(self.fit_button_axis, "Пересчет")
        self.new_frame_button.on_clicked(self._on_new_frame_clicked)
        self.reset_geom_button.on_clicked(self._on_reset_geom_clicked)
        self.fit_button.on_clicked(self._on_fit_clicked)

    def _params(self):
        width = int(self.sliders["width"].val)
        height = int(self.sliders["height"].val)
        return {
            "width": width,
            "height": height,
            "x0": min(self.sliders["x0"].val, width - 1),
            "y0": min(self.sliders["y0"].val, height - 1),
            "sigma": self.sliders["sigma"].val,
            "amplitude_lsb": self.sliders["amplitude_lsb"].val,
            "background_lsb": self.sliders["background_lsb"].val,
            "temporal_noise_lsb": self.sliders["temporal_noise_lsb"].val,
            "geometric_noise_lsb": self.sliders["geometric_noise_lsb"].val,
            "adc_bits": self.config["ADC_BITS"],
            "lsb_per_picowatt": self.config["LSB_PER_PICOWATT"],
        }

    def update(self, run_fit=False):
        params = self._params()
        self.last_frame = self.simulator.simulate(
            params["width"], params["height"], params["x0"], params["y0"], params["sigma"],
            params["amplitude_lsb"], params["background_lsb"], params["temporal_noise_lsb"],
            params["geometric_noise_lsb"], self.fix_geometric_noise, params["adc_bits"],
        )
        self.last_roi, max_x, max_y = crop_around_max(self.last_frame, 3)
        if run_fit or self.last_fit is None:
            self.last_fit = fit_gaussian_weighted(normalize_pixels_sum1(self.last_roi))
        self._draw(params, max_x, max_y)

    def _draw(self, params, max_x, max_y):
        for axis in (self.frame_axis, self.roi_axis, self.fit_axis):
            axis.clear()

        self.frame_axis.imshow(self.last_frame, cmap="gray", vmin=0, vmax=2**params["adc_bits"] - 1)
        self.frame_axis.plot(params["x0"], params["y0"], "ro", markersize=3)
        self.frame_axis.set_title("Зашумленный кадр, LSB")

        self.roi_axis.imshow(self.last_roi, cmap="gray")
        self.roi_axis.set_title("Обрезка 3×3, LSB")

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
        temporal_watts = float(lsb_to_watts(params["temporal_noise_lsb"], params["lsb_per_picowatt"]))
        self.info_axis.clear()
        self.info_axis.axis("off")
        self.info_axis.text(
            0.01,
            0.5,
            f"Кадр {self.simulator.frame_index}; максимум x={max_x}, y={max_y}; "
            f"амплитуда={amp_watts:.3e} Вт; временной шум σ={temporal_watts:.3e} Вт; "
            f"геом. шум {'фиксирован' if self.fix_geometric_noise else 'меняется'}",
            va="center",
            fontsize=10,
        )
        self.figure.canvas.draw_idle()

    def _on_slider_changed(self, value):
        self.last_fit = None
        self.update(run_fit=True)

    def _on_lsb_changed(self, value):
        self.config["LSB_PER_PICOWATT"] = float(value)
        self.update(run_fit=False)

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