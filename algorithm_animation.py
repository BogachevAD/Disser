"""Совместная пошаговая визуализация квадрантов и Нелдера–Мида.

Диалог получает уже рассчитанный fit и ничего не переоценивает. Для составного
метода квадрантная предобработка постоянно показана слева, а сохранённые
состояния Нелдера–Мида переключаются и воспроизводятся справа.
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
    """Показывает реперные итерации последнего расчёта в отдельном окне.

    fit_result содержит fit_signal, quadrant и optimization_trace; parent —
    главное окно. При открытии показана первая итерация, а таймер стоит на паузе.
    """

    def __init__(self, fit_result, parent=None):
        """Сохраняет результат fit, создаёт плеер и рисует первый кадр.

        fit_result не изменяется и повторно не вычисляется; parent задаёт владельца
        диалога. Воспроизведение начинается только после нажатия кнопки старта.
        """
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
        self.resize(1080, 680)
        self._build_ui()
        self._draw_frame()

    def _build_frames(self):
        """Формирует кадры только из сохранённых состояний Нелдера–Мида.

        Квадрантный этап не становится отдельным кадром: для составного метода он
        остаётся слева на всём протяжении правой анимации. Пустая трасса даёт None.
        """
        return list(enumerate(self.trace)) or [(0, None)]

    def _build_ui(self):
        """Создаёт две панели, пояснение и управление в стиле плеера.

        Кнопки выполняют переход к началу, шаг назад/вперёд, старт/паузу и закрытие;
        созданные виджеты сохраняются в self для синхронизации их состояния.
        """
        layout = QVBoxLayout(self)
        self.figure = Figure(figsize=(10.8, 5.4), tight_layout=True)
        self.canvas = FigureCanvas(self.figure)
        self.left_axis, self.right_axis = self.figure.subplots(1, 2)
        layout.addWidget(self.canvas, stretch=1)

        self.stage_label = QLabel()
        self.stage_label.setWordWrap(True)
        layout.addWidget(self.stage_label)

        controls = QHBoxLayout()
        controls.addStretch(1)
        self.first_button = QPushButton("⏮ Сначала")
        self.first_button.setToolTip("Перейти к первой сохранённой итерации")
        self.first_button.clicked.connect(self._go_to_first)
        controls.addWidget(self.first_button)
        self.previous_button = QPushButton("◀")
        self.previous_button.setToolTip("Предыдущая итерация")
        self.previous_button.clicked.connect(self._go_previous)
        controls.addWidget(self.previous_button)
        self.play_button = QPushButton("▶ Старт")
        self.play_button.setToolTip("Запустить или приостановить анимацию")
        self.play_button.clicked.connect(self._toggle_animation)
        controls.addWidget(self.play_button)
        self.next_button = QPushButton("▶")
        self.next_button.setToolTip("Следующая итерация")
        self.next_button.clicked.connect(self._go_next)
        controls.addWidget(self.next_button)
        self.close_button = QPushButton("Закрыть")
        self.close_button.clicked.connect(self.accept)
        controls.addWidget(self.close_button)
        controls.addStretch(1)
        layout.addLayout(controls)

    def _draw_signal(self, axis, title):
        """Рисует измеренный ROI на axis с пиксельной сеткой.

        axis — Matplotlib Axes; title подписывает панель. Одинаковая signal
        используется на всех итерациях, поэтому меняется только оценка параметров.
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

    def _draw_quadrants(self, axis):
        """Рисует квадрантную предобработку на переданной левой axis.

        Цвет показывает выбранные квадранты, числа — четыре перекрывающиеся суммы,
        красный крест — старт Нелдера–Мида, вычисленный quadrant_preprocess().
        """
        self._draw_signal(axis, "Квадрантная предобработка")
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
            axis.add_patch(
                patches.Rectangle(
                    (x, y), rectangle_width, rectangle_height,
                    edgecolor="lime" if is_selected else "orange",
                    facecolor="lime" if is_selected else "none",
                    alpha=0.18 if is_selected else 0.8,
                    linewidth=2.4 if is_selected else 1.0,
                )
            )
            axis.text(
                x + rectangle_width / 2, y + rectangle_height / 2,
                f"{name}\n{self.quadrant['sums'][name]:.3f}",
                color="lime" if is_selected else "orange", ha="center", va="center",
                fontsize=10, fontweight="bold",
            )
        axis.plot(
            self.quadrant["x0_init"], self.quadrant["y0_init"],
            marker="x", color="red", markersize=10, mew=2,
        )

    def _draw_nelder_input(self, axis, state, trace_index):
        """Рисует ROI и текущую лучшую точку обычного Нелдера–Мида.

        axis получает изображение, state содержит x0/y0/sigma, trace_index задаёт
        номер реперного состояния. Метод используется, когда квадрантного этапа нет.
        """
        self._draw_signal(axis, f"ROI: итерация Нелдера–Мида {trace_index + 1}/{len(self.frames)}")
        axis.plot(state["x0"], state["y0"], "rx", markersize=10, mew=2)
        axis.add_patch(
            patches.Circle(
                (state["x0"], state["y0"]), state["sigma"],
                edgecolor="red", facecolor="none", linestyle="--", linewidth=1.5,
            )
        )

    def _draw_nelder_model(self, axis, state, trace_index):
        """Рисует справа модель ФРТ для текущего состояния Нелдера–Мида.

        axis — правая панель; state задаёт x0/y0/sigma/loss/iteration, а
        trace_index используется в заголовке прогресса по сохранённым состояниям.
        """
        model = model_image(self.signal.shape, state["x0"], state["y0"], state["sigma"])
        model *= np.sum(self.signal)
        axis.imshow(model, cmap="gray", interpolation="nearest")
        axis.plot(state["x0"], state["y0"], "rx", markersize=10, mew=2)
        axis.add_patch(
            patches.Circle(
                (state["x0"], state["y0"]), state["sigma"],
                edgecolor="red", facecolor="none", linestyle="--", linewidth=1.5,
            )
        )
        axis.set_xticks(range(self.signal.shape[1]))
        axis.set_yticks(range(self.signal.shape[0]))
        axis.set_title(f"Нелдер–Мид: состояние {trace_index + 1}/{len(self.frames)}")

    def _draw_frame(self):
        """Перерисовывает обе панели для текущего frame_index.

        При наличии quadrant левая панель остаётся квадрантной, а правая меняется
        по trace; для чистого Нелдера–Мида слева показан ROI с текущей оценкой.
        """
        self.left_axis.clear()
        self.right_axis.clear()
        trace_index, state = self.frames[self.frame_index]
        if state is None:
            self.left_axis.axis("off")
            self.right_axis.axis("off")
            self.stage_label.setText("Для текущего расчёта нет сохранённых итераций.")
        else:
            if self.quadrant is not None:
                self._draw_quadrants(self.left_axis)
            else:
                self._draw_nelder_input(self.left_axis, state, trace_index)
            self._draw_nelder_model(self.right_axis, state, trace_index)
            final = self.frame_index == len(self.frames) - 1
            prefix = "Финальное решение" if final else "Промежуточная лучшая точка"
            quadrant_text = ""
            if self.quadrant is not None:
                selected_names = ", ".join(
                    QUADRANT_LABELS[name] for name in self.quadrant["selected_quadrants"]
                )
                quadrant_text = (
                    f"Слева: выбраны {selected_names}; Δ/Σ="
                    f"({self.quadrant['delta_x']:+.3f}, {self.quadrant['delta_y']:+.3f}), "
                    f"уверенность={self.quadrant['confidence']:.3f}. "
                )
            self.stage_label.setText(
                quadrant_text
                + f"Справа — {prefix}: итерация {state['iteration']}, "
                  f"x₀={state['x0']:.5f}, y₀={state['y0']:.5f}, "
                  f"σ={state['sigma']:.5f}, J={state['loss']:.3e}. "
                  "Показаны реперные состояния, а не все операции симплекса."
            )
        self.figure.suptitle(
            f"Работа алгоритма — итерация {self.frame_index + 1} из {len(self.frames)}",
            fontsize=13,
        )
        self._update_player_controls()
        self.canvas.draw_idle()

    def _update_player_controls(self):
        """Синхронизирует доступность кнопок с текущей позицией плеера.

        Входных переменных нет; first/previous блокируются в начале, next — в конце,
        а старт недоступен, если оптимизатор сохранил только одно состояние.
        """
        at_start = self.frame_index == 0
        at_end = self.frame_index >= len(self.frames) - 1
        self.first_button.setEnabled(not at_start)
        self.previous_button.setEnabled(not at_start)
        self.next_button.setEnabled(not at_end)
        self.play_button.setEnabled(len(self.frames) > 1)

    def _pause_animation(self):
        """Останавливает таймер и переводит общую кнопку в состояние «Старт».

        Входных аргументов нет; текущий frame_index и изображение не изменяются.
        """
        self.timer.stop()
        self.play_button.setText("▶ Старт")

    def _advance_frame(self):
        """Переходит к следующему кадру таймера и останавливается в конце.

        Входных аргументов нет; frame_index увеличивается на один, после последней
        итерации таймер выключается и плеер остаётся на финальном результате.
        """
        if self.frame_index >= len(self.frames) - 1:
            self._pause_animation()
            return
        self.frame_index += 1
        self._draw_frame()
        if self.frame_index >= len(self.frames) - 1:
            self._pause_animation()

    def _start_animation(self):
        """Запускает воспроизведение с текущей или первой итерации.

        Если плеер находится в конце, frame_index сначала сбрасывается в ноль;
        затем запускается QTimer, а единственная кнопка меняется на «Пауза».
        """
        if len(self.frames) <= 1:
            return
        if self.frame_index >= len(self.frames) - 1:
            self.frame_index = 0
            self._draw_frame()
        self.timer.start()
        self.play_button.setText("⏸ Пауза")

    def _go_to_first(self):
        """Ставит воспроизведение на паузу и показывает первую итерацию.

        Входных аргументов нет; действие соответствует кнопке «Сначала» плеера.
        """
        self._pause_animation()
        self.frame_index = 0
        self._draw_frame()

    def _go_previous(self):
        """Ставит плеер на паузу и выполняет один шаг влево.

        Индекс ограничивается нулём, поэтому повторное нажатие в начале безопасно.
        """
        self._pause_animation()
        self.frame_index = max(0, self.frame_index - 1)
        self._draw_frame()

    def _go_next(self):
        """Ставит плеер на паузу и выполняет один шаг вправо.

        Индекс ограничивается последним кадром, поэтому выйти за trace невозможно.
        """
        self._pause_animation()
        self.frame_index = min(len(self.frames) - 1, self.frame_index + 1)
        self._draw_frame()

    def _toggle_animation(self):
        """Переключает единственную кнопку между стартом и паузой.

        Активный timer останавливается; при паузе запускается продолжение, а из
        последней итерации воспроизведение автоматически начинается сначала.
        """
        if self.timer.isActive():
            self._pause_animation()
        else:
            self._start_animation()
