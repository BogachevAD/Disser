"""Всплывающая анимация квадрантной предобработки и Нелдера–Мида.

Диалог получает уже рассчитанный fit и ничего не переоценивает. Он последовательно
показывает выбранные квадранты, сокращённую трассу лучших точек оптимизатора и
соответствующие модели ФРТ, поэтому визуализация воспроизводима для данного кадра.
"""

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout

# Qt импортируется до backend_qtagg, чтобы PyInstaller однозначно выбрал PyQt6.
import matplotlib.patches as patches
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from gaussian_math import model_image


QUADRANT_LABELS = {
    "LT": "левый верхний",
    "RT": "правый верхний",
    "LB": "левый нижний",
    "RB": "правый нижний",
}


class AlgorithmAnimationDialog(QDialog):
    """Показывает этапы последнего расчёта в отдельном PyQt6-окне.

    fit_result содержит fit_signal, quadrant и optimization_trace; parent —
    главное окно. Таймер перебирает только сохранённые реперные состояния.
    """

    def __init__(self, fit_result, parent=None):
        super().__init__(parent)
        self.fit_result = fit_result
        self.signal = np.asarray(fit_result["fit_signal"], dtype=float)
        self.quadrant = fit_result.get("quadrant")
        self.trace = list(fit_result.get("optimization_trace", []))
        self.frames = self._build_frames()
        self.frame_index = 0
        self.timer = QTimer(self)
        self.timer.setInterval(750)
        self.timer.timeout.connect(self._advance_frame)

        self.setWindowTitle("Как работает выбранный алгоритм")
        self.resize(1000, 620)
        self._build_ui()
        self._draw_frame()
        QTimer.singleShot(300, self._start_animation)

    def _build_frames(self):
        """Формирует последовательность этапов из fit_result.

        Квадрантный кадр добавляется только для комбинированного метода; затем
        идут сохранённые состояния Нелдера–Мида от начального к финальному.
        """
        frames = []
        if self.quadrant is not None:
            frames.append(("quadrant", self.quadrant))
        frames.extend(("nelder_mead", (index, state)) for index, state in enumerate(self.trace))
        return frames or [("empty", None)]

    def _build_ui(self):
        """Создаёт холст, строку пояснения и кнопки управления.

        Входных аргументов нет; созданные виджеты сохраняются в полях диалога.
        """
        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(10, 5), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.input_axis, self.model_axis = self.figure.subplots(1, 2)
        layout.addWidget(self.canvas, stretch=1)

        self.stage_label = QLabel()
        self.stage_label.setWordWrap(True)
        layout.addWidget(self.stage_label)

        controls = QHBoxLayout()
        controls.addStretch(1)
        self.restart_button = QPushButton("Сначала")
        self.restart_button.clicked.connect(self._restart_animation)
        controls.addWidget(self.restart_button)
        self.play_button = QPushButton("Пауза")
        self.play_button.clicked.connect(self._toggle_animation)
        controls.addWidget(self.play_button)
        self.close_button = QPushButton("Закрыть")
        self.close_button.clicked.connect(self.accept)
        controls.addWidget(self.close_button)
        layout.addLayout(controls)

    def _draw_signal(self, axis, title):
        """Рисует измеренный ROI на axis с пиксельной сеткой.

        axis — Matplotlib Axes; title подписывает текущий этап. Общая шкала
        сигнала сохраняется на всех кадрах анимации.
        """
        axis.imshow(self.signal, cmap="gray", interpolation="nearest")
        height, width = self.signal.shape
        axis.set_xticks(np.arange(-0.5, width, 1), minor=True)
        axis.set_yticks(np.arange(-0.5, height, 1), minor=True)
        axis.grid(which="minor", color="cyan", linewidth=0.45, alpha=0.55)
        axis.tick_params(which="minor", bottom=False, left=False)
        axis.set_xticks(range(width))
        axis.set_yticks(range(height))
        axis.set_title(title)

    def _draw_quadrants(self):
        """Отображает четыре суммы и выделяет максимальные квадранты.

        Данные берутся из self.quadrant; перекрытие по центральной строке и
        столбцу соответствует математической функции quadrant_preprocess().
        """
        self._draw_signal(self.input_axis, "ROI и четыре квадрантные суммы")
        height, width = self.signal.shape
        center_y, center_x = height // 2, width // 2
        bounds = {
            "LT": (-0.5, -0.5, center_x + 1, center_y + 1),
            "RT": (center_x - 0.5, -0.5, width - center_x, center_y + 1),
            "LB": (-0.5, center_y - 0.5, center_x + 1, height - center_y),
            "RB": (center_x - 0.5, center_y - 0.5, width - center_x, height - center_y),
        }
        selected = set(self.quadrant["selected_quadrants"])
        for name, (x, y, rectangle_width, rectangle_height) in bounds.items():
            is_selected = name in selected
            self.input_axis.add_patch(
                patches.Rectangle(
                    (x, y), rectangle_width, rectangle_height,
                    edgecolor="lime" if is_selected else "orange",
                    facecolor="lime" if is_selected else "none",
                    alpha=0.18 if is_selected else 0.8,
                    linewidth=2.4 if is_selected else 1.0,
                )
            )
            self.input_axis.text(
                x + rectangle_width / 2, y + rectangle_height / 2,
                f"{name}\n{self.quadrant['sums'][name]:.3f}",
                color="lime" if is_selected else "orange", ha="center", va="center",
                fontsize=10, fontweight="bold",
            )
        self.input_axis.plot(
            self.quadrant["x0_init"], self.quadrant["y0_init"],
            marker="x", color="red", markersize=10, mew=2,
        )

        self.model_axis.axis("off")
        selected_codes = " / ".join(self.quadrant["selected_quadrants"])
        selected_names = ", ".join(
            QUADRANT_LABELS[name] for name in self.quadrant["selected_quadrants"]
        )
        if len(self.quadrant["selected_quadrants"]) > 2:
            selected_names = "\n".join(
                QUADRANT_LABELS[name] for name in self.quadrant["selected_quadrants"]
            )
        self.model_axis.text(
            0.03, 0.95,
            "Квадрантная предобработка\n\n"
            f"Выбрано: {selected_codes}\n"
            f"({selected_names})\n"
            f"Δx/Σ = {self.quadrant['delta_x']:+.4f}\n"
            f"Δy/Σ = {self.quadrant['delta_y']:+.4f}\n"
            f"Уверенность = {self.quadrant['confidence']:.4f}\n\n"
            f"Старт Нелдера–Мида:\n"
            f"x₀ = {self.quadrant['x0_init']:.4f}\n"
            f"y₀ = {self.quadrant['y0_init']:.4f}\n"
            f"Предполагаемый пиксель = "
            f"({self.quadrant['coarse_pixel_x']}, {self.quadrant['coarse_pixel_y']})",
            transform=self.model_axis.transAxes, va="top", fontsize=12,
        )
        self.stage_label.setText(
            "Сначала сравниваются суммы четырёх перекрывающихся областей. "
            "Красный крест задаёт стартовую координату; это ещё не итоговый fit."
        )

    def _draw_nelder_mead(self, state, trace_index):
        """Рисует одно сохранённое состояние оптимизатора.

        state содержит iteration, x0, y0, sigma и loss; trace_index нужен для
        подписи прогресса. Справа строится соответствующая нормированная ФРТ.
        """
        total_states = max(len(self.trace), 1)
        self._draw_signal(
            self.input_axis,
            f"Этап 2. Нелдер–Мид: состояние {trace_index + 1}/{total_states}",
        )
        self.input_axis.plot(state["x0"], state["y0"], "rx", markersize=10, mew=2)
        self.input_axis.add_patch(
            patches.Circle(
                (state["x0"], state["y0"]), state["sigma"],
                edgecolor="red", facecolor="none", linestyle="--", linewidth=1.5,
            )
        )

        model = model_image(self.signal.shape, state["x0"], state["y0"], state["sigma"])
        model *= np.sum(self.signal)
        self.model_axis.imshow(model, cmap="gray", interpolation="nearest")
        self.model_axis.plot(state["x0"], state["y0"], "rx", markersize=10, mew=2)
        self.model_axis.set_xticks(range(self.signal.shape[1]))
        self.model_axis.set_yticks(range(self.signal.shape[0]))
        self.model_axis.set_title("Модель в текущей лучшей точке")
        final = trace_index == len(self.trace) - 1
        self.stage_label.setText(
            f"{'Финальное решение' if final else 'Промежуточная лучшая точка'}: "
            f"итерация {state['iteration']}, x₀={state['x0']:.5f}, "
            f"y₀={state['y0']:.5f}, σ={state['sigma']:.5f}, J={state['loss']:.3e}. "
            "Показаны реперные состояния, а не все операции с вершинами симплекса."
        )

    def _draw_frame(self):
        """Перерисовывает текущий элемент self.frames.

        frame_index выбирает квадрантный, оптимизационный или пустой этап;
        после отрисовки холст обновляется без блокировки Qt event loop.
        """
        self.input_axis.clear()
        self.model_axis.clear()
        stage, payload = self.frames[self.frame_index]
        if stage == "quadrant":
            self._draw_quadrants()
        elif stage == "nelder_mead":
            trace_index, state = payload
            self._draw_nelder_mead(state, trace_index)
        else:
            self.input_axis.axis("off")
            self.model_axis.axis("off")
            self.stage_label.setText("Для текущего расчёта нет сохранённых этапов.")
        self.figure.suptitle(
            f"Работа алгоритма — кадр {self.frame_index + 1} из {len(self.frames)}",
            fontsize=13,
        )
        self.canvas.draw_idle()

    def _advance_frame(self):
        """Переходит к следующему кадру таймера или останавливается в конце.

        Входных аргументов нет; меняются frame_index, timer и подпись play_button.
        """
        if self.frame_index >= len(self.frames) - 1:
            self.timer.stop()
            self.play_button.setText("Повторить")
            return
        self.frame_index += 1
        self._draw_frame()

    def _start_animation(self):
        """Запускает таймер, если в последовательности больше одного кадра.

        Функция вызывается после показа окна и обновляет подпись кнопки паузы.
        """
        if len(self.frames) > 1:
            self.timer.start()
            self.play_button.setText("Пауза")

    def _restart_animation(self):
        """Возвращает анимацию к первому этапу и запускает её заново.

        Входных аргументов нет; старый таймер безопасно перезапускается.
        """
        self.timer.stop()
        self.frame_index = 0
        self._draw_frame()
        self._start_animation()

    def _toggle_animation(self):
        """Переключает воспроизведение между паузой, продолжением и повтором.

        Текущее состояние QTimer и последний frame_index определяют действие.
        """
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText("Продолжить")
        elif self.frame_index >= len(self.frames) - 1:
            self._restart_animation()
        else:
            self._start_animation()
