"""Регрессионные тесты математического ядра и генератора кадров.

Каждый тест фиксирует инженерный контракт: точность интеграла, восстановление
ФРТ, выбор ROI при шумовой помехе, координатную привязку и пересчёт единиц.
"""

import unittest

import numpy as np

from gaussian_app import GaussianFrameSimulator
from gaussian_math import (
    FIT_METHOD_NELDER_MEAD,
    crop_around_detected_target,
    crop_around_max,
    crop_around_pixel,
    crop_around_position,
    estimate_background_ring,
    fit_gaussian,
    fit_gaussian_weighted,
    gaussian_pixel_integral,
    local_to_global,
    lsb_to_watts,
    model_image,
    watts_to_lsb,
)


class GaussianMathTests(unittest.TestCase):
    """Проверяет численную корректность публичных функций.

    Каждый метод подготавливает входные переменные и проверяет один контракт.
    """

    def test_vectorized_model_matches_scalar_pixel_integrals(self):
        """Сравнивает векторную ФРТ с интегралами пикселей.

        shape, x0, y0 и sigma задают один контрольный несимметричный случай.
        """
        shape = (5, 7)
        x0, y0, sigma = 2.35, 3.1, 0.73
        expected = np.array(
            [
                [gaussian_pixel_integral(x0, y0, sigma, x, y) for x in range(shape[1])]
                for y in range(shape[0])
            ]
        )
        expected /= expected.sum()
        np.testing.assert_allclose(model_image(shape, x0, y0, sigma), expected, rtol=1e-13, atol=1e-15)

    def test_fit_recovers_ideal_subpixel_gaussian(self):
        """Проверяет fit на трёх идеальных ROI.

        expected содержит истинные local x0, local y0 и sigma.
        """
        for expected in ((1.0, 1.0, 0.63), (1.25, 0.8, 1.0), (0.65, 1.35, 1.5)):
            with self.subTest(expected=expected):
                result = fit_gaussian_weighted(model_image((3, 3), *expected))
                self.assertTrue(result["success"])
                np.testing.assert_allclose(
                    [result["x0"], result["y0"], result["sigma"]], expected, rtol=0, atol=2e-5
                )

    def test_fixed_geometric_pattern_scales_with_requested_sigma(self):
        """Проверяет масштаб фиксированной карты.

        Два кадра используют один seed, но sigma шума 2 и 5 LSB.
        """
        simulator = GaussianFrameSimulator(rng=np.random.default_rng(42))
        common = (8, 6, 3.0, 2.0, 0.8, 0.0, 10_000.0, 0.0)
        frame_sigma_2 = simulator.simulate(*common, 2.0, True, 16)
        frame_sigma_5 = simulator.simulate(*common, 5.0, True, 16)
        np.testing.assert_allclose(frame_sigma_5 - 10_000.0, 2.5 * (frame_sigma_2 - 10_000.0))

    def test_crop_at_edge_keeps_requested_shape(self):
        """Проверяет ROI около верхнего левого края.

        image имеет максимум (0,0), а crop обязан сохранить размер 3×3.
        """
        image = np.zeros((4, 4))
        image[0, 0] = 1.0
        crop, x_max, y_max = crop_around_max(image, 3)
        self.assertEqual(crop.shape, (3, 3))
        self.assertEqual((x_max, y_max), (0, 0))

    def test_crop_rejects_even_size(self):
        """Проверяет запрет чётного размера.

        size=2 не имеет единственного центрального пикселя и вызывает ValueError.
        """
        with self.assertRaises(ValueError):
            crop_around_max(np.ones((3, 3)), 2)

    def test_lsb_watt_conversion_round_trip(self):
        """Проверяет обратимость преобразования LSB↔Вт.

        values переводятся при 60 LSB/пВт; нулевой коэффициент запрещён.
        """
        values = np.array([0.0, 1.0, 60.0, 30_000.0])
        np.testing.assert_allclose(watts_to_lsb(lsb_to_watts(values, 60.0), 60.0), values)
        with self.assertRaises(ValueError):
            lsb_to_watts(values, 0.0)

    def test_truth_roi_is_independent_of_noise_maximum(self):
        """Проверяет независимость truth ROI от выброса.

        Яркий пиксель (2,2) не должен изменить окно заданного центра (20.5,16).
        """
        image = np.zeros((32, 32))
        image[2, 2] = 1_000_000.0
        selection = crop_around_position(image, 20.5, 16.0, 3)
        self.assertEqual((selection.center_x, selection.center_y), (21, 16))
        self.assertEqual((selection.origin_x, selection.origin_y), (20, 15))

    def test_matched_filter_rejects_isolated_bright_pixel(self):
        """Сравнивает сырой максимум и согласованный фильтр.

        Выброс ярче пика, но распределённое пятно имеет больший отклик фильтра.
        """
        simulator = GaussianFrameSimulator(rng=np.random.default_rng(1))
        frame = simulator.simulate(32, 32, 16.2, 15.8, 1.0, 100.0, 0.0, 0.0, 0.0, True, 16)
        frame[2, 2] = 110.0
        self.assertEqual(np.unravel_index(np.argmax(frame), frame.shape), (2, 2))
        selection = crop_around_detected_target(frame, sigma=1.0, background=0.0, size=3)
        self.assertEqual((selection.center_x, selection.center_y), (16, 16))

    def test_global_fit_is_stable_for_either_half_pixel_roi(self):
        """Проверяет глобальную привязку при x=n+0.5.

        ROI вокруг пикселей 20 и 21 должны восстановить один центр (20.5,16).
        """
        simulator = GaussianFrameSimulator(rng=np.random.default_rng(2))
        frame = simulator.simulate(32, 32, 20.5, 16.0, 1.0, 30_000.0, 0.0, 0.0, 0.0, True, 16)
        estimates = []
        for center_x in (20, 21):
            roi, origin_x, origin_y = crop_around_pixel(frame, center_x, 16, 3)
            fit = fit_gaussian_weighted(roi)
            estimates.append(local_to_global(fit["x0"], fit["y0"], origin_x, origin_y))
        np.testing.assert_allclose(estimates, [(20.5, 16.0), (20.5, 16.0)], atol=2e-5)

    def test_background_ring_excludes_roi_and_guard_gap(self):
        """Проверяет геометрию фоновой рамки.

        Сигнал в ROI и защитном отступе не должен влиять на uniform background.
        """
        image = np.full((15, 15), 1_000.0)
        image[4:11, 4:11] = 50_000.0
        statistics = estimate_background_ring(image, 7, 7, roi_size=3, ring_width=1, ring_gap=2)
        self.assertEqual(statistics.pixel_count, 32)
        self.assertEqual(statistics.mean, 1_000.0)
        self.assertEqual(statistics.std, 0.0)

    def test_nelder_mead_background_checkbox_changes_preprocessing(self):
        """Проверяет Нелдер–Мид с вычитанием фона и без него.

        Известный фон должен дать точную sigma; невычтенный фон ожидаемо смещает её.
        """
        signal = 5_000.0 * model_image((3, 3), 1.2, 0.8, 0.8)
        roi = 1_000.0 + signal
        corrected = fit_gaussian(
            roi, FIT_METHOD_NELDER_MEAD, background_level=1_000.0,
            subtract_background=True, noise_sigma=20.0,
        )
        raw = fit_gaussian(
            roi, FIT_METHOD_NELDER_MEAD, background_level=1_000.0,
            subtract_background=False, noise_sigma=20.0,
        )
        without_noise_scale = fit_gaussian(
            roi, FIT_METHOD_NELDER_MEAD, background_level=1_000.0,
            subtract_background=True, noise_sigma=None,
        )
        self.assertEqual(corrected["method"], FIT_METHOD_NELDER_MEAD)
        np.testing.assert_allclose(
            [corrected["x0"], corrected["y0"], corrected["sigma"]],
            [1.2, 0.8, 0.8], atol=2e-5,
        )
        self.assertGreater(abs(raw["sigma"] - 0.8), 0.1)
        np.testing.assert_allclose(
            [without_noise_scale["x0"], without_noise_scale["y0"], without_noise_scale["sigma"]],
            [corrected["x0"], corrected["y0"], corrected["sigma"]], atol=1e-10,
        )
        self.assertTrue(np.isnan(without_noise_scale["reduced_chi_square"]))


if __name__ == "__main__":
    unittest.main()
