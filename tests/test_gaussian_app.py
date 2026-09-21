"""Интеграционные проверки интерфейса фиксации и выбора шумовых карт.

Qt запускается в offscreen-режиме: тесты проверяют сигналы виджетов и полный
пересчёт модели без открытия окна на рабочем столе.
"""

import os
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication, QGroupBox, QTableWidget, QToolButton

from algorithm_animation import AlgorithmAnimationDialog
from gaussian_app import GaussianSimulatorWindow
from run_gauss_simulator import CONFIG


class GaussianAppNoiseTests(unittest.TestCase):
    """Проверяет пользовательский сценарий сравнения на одном шумовом кадре.

    Общий QApplication создаётся один раз; каждое окно получает отдельный config
    с ненулевыми шумами и закрывается после теста.
    """

    @classmethod
    def setUpClass(cls):
        """Создаёт offscreen QApplication для всех тестов класса.

        Входных параметров нет; экземпляр сохраняется в cls.app и не запускает
        event loop, поскольку Qt-сигналы spinbox/combobox синхронны.
        """
        cls.app = QApplication.instance() or QApplication([])

    def make_window(self):
        """Возвращает окно с фиксированными ненулевыми шумами.

        Копия CONFIG исключает влияние теста на стартовые настройки приложения.
        """
        config = dict(CONFIG)
        config.update({
            "TEMPORAL_NOISE_LSB": 15.0,
            "GEOMETRIC_NOISE_LSB": 4.0,
            "FIX_TEMPORAL_NOISE": True,
            "FIX_GEOMETRIC_NOISE": True,
        })
        return GaussianSimulatorWindow(config)

    def test_method_and_ring_changes_keep_fixed_noise_frame(self):
        """Сравнивает входной кадр до и после смены метода и рамки.

        При двух включённых фиксациях меняется только обработка уже созданного
        кадра; массив last_frame обязан остаться побитно одинаковым.
        """
        window = self.make_window()
        try:
            initial_frame = window.last_frame.copy()
            with patch.object(
                window.simulator, "simulate", wraps=window.simulator.simulate,
            ) as simulate_mock:
                window.fit_method_combo.setCurrentIndex(1)
                np.testing.assert_array_equal(window.last_frame, initial_frame)
                window.ring_width_combo.setCurrentIndex(1)
                np.testing.assert_array_equal(window.last_frame, initial_frame)
                simulate_mock.assert_not_called()
        finally:
            window.close()

    def test_zero_noise_resize_keeps_both_histories_empty(self):
        """Проверяет интерфейс при нулевых уровнях и изменении размера кадра.

        Кнопки новых карт отключены, изменение width не создаёт seed и оба списка
        истории остаются пустыми независимо от установленных флажков фиксации.
        """
        config = dict(CONFIG)
        config.update({"TEMPORAL_NOISE_LSB": 0.0, "GEOMETRIC_NOISE_LSB": 0.0})
        window = GaussianSimulatorWindow(config)
        try:
            self.assertFalse(window.generate_temporal_button.isEnabled())
            self.assertFalse(window.generate_geom_button.isEnabled())
            window.inputs["width"].setValue(window.inputs["width"].value() + 1)
            self.assertEqual(window.simulator.temporal_noise_history, [])
            self.assertEqual(window.simulator.geometric_noise_history, [])
            self.assertIsNone(window.simulator.temporal_noise_map_id)
            self.assertIsNone(window.simulator.geometric_noise_map_id)
        finally:
            window.close()

    def test_large_frame_preview_is_decimated_without_coordinate_change(self):
        """Проверяет облегчённую отрисовку больших кадров.

        Preview содержит меньше элементов, но extent остаётся в исходных индексах,
        поэтому маркеры центра, ROI и фоновой рамки не получают смещения.
        """
        window = self.make_window()
        try:
            frame = np.zeros((1200, 800), dtype=float)
            frame[1, 1] = 5.0
            preview, extent, stride = window._frame_for_display(frame)
            self.assertEqual(stride, 2)
            self.assertEqual(preview.shape, (600, 400))
            self.assertEqual(preview[0, 0], 5.0)
            self.assertEqual(extent, (-0.5, 799.5, 1199.5, -0.5))
        finally:
            window.close()

    def test_previous_temporal_map_can_be_selected_from_combo(self):
        """Создаёт новую карту и затем возвращает первую через выпадающий список.

        Выбор предыдущей map_id должен восстановить исходный полный кадр и оставить
        флажок временной фиксации включённым.
        """
        window = self.make_window()
        try:
            initial_id = window.simulator.temporal_noise_map_id
            self.assertEqual(initial_id, 0)
            self.assertEqual(window.simulator.geometric_noise_map_id, 0)
            self.assertTrue(window.temporal_noise_combo.currentText().startswith("T#0"))
            self.assertTrue(window.geometric_noise_combo.currentText().startswith("G#0"))
            initial_frame = window.last_frame.copy()
            window.generate_temporal_button.click()
            self.assertEqual(window.simulator.temporal_noise_map_id, 1)
            self.assertEqual(window.simulator.geometric_noise_map_id, 0)
            previous_index = window.temporal_noise_combo.findData(initial_id)
            self.assertGreaterEqual(previous_index, 0)
            window.temporal_noise_combo.setCurrentIndex(previous_index)
            self.assertTrue(window.fix_temporal_checkbox.isChecked())
            np.testing.assert_array_equal(window.last_frame, initial_frame)
        finally:
            window.close()

    def test_matrix_tables_and_animation_button_layout(self):
        """Проверяет замену текстовых полей таблицами и положение кнопки.

        Три матрицы должны быть QTableWidget нужного размера, а кнопка анимации —
        дочерним элементом группы выбора ROI, фона и алгоритма.
        """
        window = self.make_window()
        try:
            for table in (window.roi_matrix, window.fit_matrix, window.model_matrix):
                self.assertIsInstance(table, QTableWidget)
                self.assertEqual(table.rowCount(), 3)
                self.assertEqual(table.columnCount(), 3)
                self.assertIsNotNone(table.item(0, 0))
            button_group = window.animation_button.parentWidget()
            self.assertIsInstance(button_group, QGroupBox)
            self.assertEqual(button_group.title(), "ROI, фон и оценивание")
            self.assertEqual(window.fit_method_combo.count(), 4)
            self.assertIn("Робастный", window.fit_method_combo.itemText(2))
            self.assertIn("робастный", window.fit_method_combo.itemText(3))
        finally:
            window.close()

    def test_combined_animation_is_side_by_side_and_initially_paused(self):
        """Проверяет совместный экран квадрантов и Нелдера–Мида.

        Слева всегда остаются квадранты, справа показана текущая итерация trace;
        стрелки переключают кадры, а одна кнопка запускает и ставит на паузу.
        """
        window = self.make_window()
        dialog = None
        try:
            window.fit_method_combo.setCurrentIndex(1)
            dialog = AlgorithmAnimationDialog(window.last_fit, window)
            self.assertFalse(dialog.timer.isActive())
            self.assertEqual(dialog.play_button.text(), "▶ Старт")
            self.assertEqual(len(dialog.frames), len(window.last_fit["optimization_trace"]))
            self.assertIn("Квадрантная", dialog.left_axis.get_title())
            self.assertIn("Нелдер–Мид", dialog.right_axis.get_title())

            if len(dialog.frames) > 1:
                dialog.next_button.click()
                self.assertEqual(dialog.frame_index, 1)
                self.assertFalse(dialog.timer.isActive())
                dialog.previous_button.click()
                self.assertEqual(dialog.frame_index, 0)
                dialog.play_button.click()
                self.assertTrue(dialog.timer.isActive())
                self.assertEqual(dialog.play_button.text(), "⏸ Пауза")
                dialog.play_button.click()
                self.assertFalse(dialog.timer.isActive())
                self.assertEqual(dialog.play_button.text(), "▶ Старт")
        finally:
            if dialog is not None:
                dialog.close()
            window.close()

    def test_help_button_opens_embedded_method_documentation(self):
        """Проверяет значок справки и загрузку полного текста из README.

        Клик не должен пересчитывать модель; диалог содержит разделы про
        классический функционал, Huber и прямую сравнительную таблицу.
        """
        window = self.make_window()
        try:
            calculation_index = window.calculation_index
            self.assertIsInstance(window.help_button, QToolButton)
            self.assertEqual(
                window.help_button.accessibleName(), "Справка по методам оценивания",
            )
            window.help_button.click()
            self.app.processEvents()
            dialog = window.method_help_dialog
            self.assertIsNotNone(dialog)
            self.assertTrue(dialog.isVisible())
            help_text = dialog.browser.toPlainText()
            self.assertIn("Классический функционал", help_text)
            self.assertIn("Функция потерь Хьюбера", help_text)
            self.assertIn("Прямое сравнение реализаций", help_text)
            self.assertEqual(window.calculation_index, calculation_index)
            dialog.close()
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
