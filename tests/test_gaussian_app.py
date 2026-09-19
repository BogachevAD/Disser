"""Интеграционные проверки интерфейса фиксации и выбора шумовых карт.

Qt запускается в offscreen-режиме: тесты проверяют сигналы виджетов и полный
пересчёт модели без открытия окна на рабочем столе.
"""

import os
import unittest

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

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
            window.fit_method_combo.setCurrentIndex(1)
            np.testing.assert_array_equal(window.last_frame, initial_frame)
            window.ring_width_combo.setCurrentIndex(1)
            np.testing.assert_array_equal(window.last_frame, initial_frame)
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
            initial_frame = window.last_frame.copy()
            window.generate_temporal_button.click()
            self.assertNotEqual(window.simulator.temporal_noise_map_id, initial_id)
            previous_index = window.temporal_noise_combo.findData(initial_id)
            self.assertGreaterEqual(previous_index, 0)
            window.temporal_noise_combo.setCurrentIndex(previous_index)
            self.assertTrue(window.fix_temporal_checkbox.isChecked())
            np.testing.assert_array_equal(window.last_frame, initial_frame)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
