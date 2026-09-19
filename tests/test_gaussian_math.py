"""Регрессионные тесты математического ядра и генератора кадров.

Каждый тест фиксирует инженерный контракт: точность интеграла, восстановление
ФРТ, выбор ROI при шумовой помехе, координатную привязку и пересчёт единиц.
"""

import unittest

import numpy as np

from gaussian_app import GaussianFrameSimulator
from gaussian_math import (
    FIT_METHOD_NELDER_MEAD,
    FIT_METHOD_QUADRANT_NELDER_MEAD,
    FIT_METHOD_QUADRANT_ROBUST_NELDER_MEAD,
    FIT_METHOD_ROBUST_NELDER_MEAD,
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
    quadrant_preprocess,
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

    def test_quadrant_preprocessor_selects_expected_corner(self):
        """Проверяет суммы квадрантов для пятна справа сверху.

        Идеальная ФРТ с центром (1.7,0.3) должна выбрать RT, дать правильные
        знаки Δ/Σ и указать дискретный пиксель (2,0).
        """
        signal = model_image((3, 3), 1.7, 0.3, 0.8)
        quadrant = quadrant_preprocess(signal)
        self.assertEqual(quadrant["selected_quadrants"], ["RT"])
        self.assertGreater(quadrant["delta_x"], 0.0)
        self.assertLess(quadrant["delta_y"], 0.0)
        self.assertEqual((quadrant["coarse_pixel_x"], quadrant["coarse_pixel_y"]), (2, 0))

    def test_quadrant_tie_does_not_bias_centered_spot(self):
        """Исключает произвольный выбор угла для центрированного пятна.

        При равенстве четырёх сумм объединяются все квадранты, поэтому стартовая
        оценка остаётся в центре ROI, а уверенность практически равна нулю.
        """
        quadrant = quadrant_preprocess(model_image((3, 3), 1.0, 1.0, 0.8))
        self.assertEqual(set(quadrant["selected_quadrants"]), {"LT", "RT", "LB", "RB"})
        np.testing.assert_allclose([quadrant["x0_init"], quadrant["y0_init"]], [1.0, 1.0], atol=1e-14)
        self.assertLess(quadrant["confidence"], 1e-12)

    def test_quadrant_then_nelder_mead_recovers_ideal_gaussian(self):
        """Проверяет полную цепочку квадранты → Нелдер–Мид.

        Результат должен восстановить заданные x0/y0/sigma, сохранить сведения
        о RT-квадранте и несколько реперных точек для анимации интерфейса.
        """
        expected = (1.7, 0.3, 0.8)
        result = fit_gaussian(
            model_image((3, 3), *expected), FIT_METHOD_QUADRANT_NELDER_MEAD,
            background_level=0.0, subtract_background=False, noise_sigma=None,
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["method"], FIT_METHOD_QUADRANT_NELDER_MEAD)
        self.assertEqual(result["quadrant"]["selected_quadrants"], ["RT"])
        self.assertGreaterEqual(len(result["optimization_trace"]), 2)
        self.assertLessEqual(len(result["optimization_trace"]), 10)
        np.testing.assert_allclose(
            [result["x0"], result["y0"], result["sigma"]], expected, atol=2e-5,
        )

    def test_quadrant_fit_is_stable_in_all_four_directions(self):
        """Проверяет комбинированный метод во всех направлениях от центра.

        Четыре несимметричных положения исключают скрытую привязку реализации к
        одному углу ROI; каждый fit должен сойтись к одной точности.
        """
        for expected in (
            (0.35, 0.55, 0.75), (1.65, 0.45, 0.9),
            (0.45, 1.7, 1.1), (1.6, 1.55, 0.65),
        ):
            with self.subTest(expected=expected):
                result = fit_gaussian(
                    model_image((3, 3), *expected), FIT_METHOD_QUADRANT_NELDER_MEAD,
                    background_level=0.0, subtract_background=False, noise_sigma=None,
                )
                self.assertTrue(result["success"])
                np.testing.assert_allclose(
                    [result["x0"], result["y0"], result["sigma"]], expected, atol=3e-5,
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

    def test_fixed_temporal_noise_reuses_exact_frame(self):
        """Проверяет воспроизводимость кадра при фиксации временного шума.

        Два полных расчёта с одинаковыми параметрами должны вернуть побитно
        одинаковый кадр и создать только одну запись временной карты.
        """
        simulator = GaussianFrameSimulator(rng=np.random.default_rng(43))
        arguments = (8, 6, 3.2, 2.1, 0.8, 100.0, 10_000.0, 12.0, 0.0, True, 16, True)
        first_frame = simulator.simulate(*arguments)
        second_frame = simulator.simulate(*arguments)
        np.testing.assert_array_equal(second_frame, first_frame)
        self.assertEqual(len(simulator.temporal_noise_history), 1)

    def test_noise_cache_keeps_ten_and_restores_by_seed(self):
        """Проверяет глубину кэша и точное восстановление старой карты.

        После 12 генераций остаются 10 последних записей; выбранный seed должен
        восстановить тот же массив N(0,1), не сохраняя его копию в истории.
        """
        simulator = GaussianFrameSimulator(rng=np.random.default_rng(44))
        for _ in range(12):
            simulator.generate_temporal_noise((4, 5))
        self.assertEqual(len(simulator.temporal_noise_history), 10)
        retained = simulator.temporal_noise_history[-1]
        expected = np.random.default_rng(retained.seed).standard_normal(retained.shape)
        self.assertTrue(simulator.select_noise_map("temporal", retained.map_id))
        np.testing.assert_array_equal(simulator.temporal_noise, expected)
        self.assertFalse(simulator.select_noise_map("temporal", -1))

    def test_noise_histories_are_independent_and_filter_by_shape(self):
        """Проверяет независимость двух кэшей и фильтрацию по размеру.

        Temporal и geometric записи не смешиваются; запрос shape возвращает
        только карты, совместимые с выбранным размером кадра.
        """
        simulator = GaussianFrameSimulator(rng=np.random.default_rng(45))
        simulator.generate_temporal_noise((3, 3))
        simulator.generate_temporal_noise((5, 7))
        simulator.generate_geometric_noise((3, 3))
        self.assertEqual(len(simulator.temporal_noise_history), 2)
        self.assertEqual(len(simulator.geometric_noise_history), 1)
        records = simulator.noise_map_records("temporal", (3, 3))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].shape, (3, 3))

    def test_noise_map_numbers_are_independent_and_cycle_from_zero_to_ten(self):
        """Проверяет отдельную нумерацию temporal и geometric в диапазоне 0–10.

        Первые карты обоих типов имеют номер 0; после 10 каждый собственный
        счётчик независимо возвращается к нулю без дубликатов внутри кэша из 10.
        """
        simulator = GaussianFrameSimulator(rng=np.random.default_rng(46))
        simulator.generate_temporal_noise((3, 3))
        simulator.generate_geometric_noise((3, 3))
        self.assertEqual(simulator.temporal_noise_map_id, 0)
        self.assertEqual(simulator.geometric_noise_map_id, 0)

        for _ in range(11):
            simulator.generate_temporal_noise((3, 3))
        self.assertEqual(simulator.temporal_noise_map_id, 0)
        temporal_ids = [record.map_id for record in simulator.temporal_noise_history]
        self.assertEqual(len(temporal_ids), len(set(temporal_ids)))
        self.assertTrue(all(0 <= map_id <= 10 for map_id in temporal_ids))
        self.assertEqual(simulator.geometric_noise_map_id, 0)

    def test_zero_noise_does_not_create_or_number_maps(self):
        """Исключает генерацию карт при нулевом СКО обоих шумов.

        Несколько размеров и оба состояния фиксации не должны менять историю,
        текущие map_id или независимые счётчики карт.
        """
        simulator = GaussianFrameSimulator(rng=np.random.default_rng(47))
        simulator.simulate(8, 6, 3.0, 2.0, 0.8, 100.0, 10.0, 0.0, 0.0, True, 16, True)
        simulator.simulate(12, 10, 3.0, 2.0, 0.8, 100.0, 10.0, 0.0, 0.0, False, 16, False)
        self.assertEqual(simulator.temporal_noise_history, [])
        self.assertEqual(simulator.geometric_noise_history, [])
        self.assertIsNone(simulator.temporal_noise_map_id)
        self.assertIsNone(simulator.geometric_noise_map_id)
        self.assertEqual(simulator.temporal_noise_map_counter, 0)
        self.assertEqual(simulator.geometric_noise_map_counter, 0)
        self.assertIsNone(simulator.generate_temporal_noise((3, 3), 0.0))
        self.assertIsNone(simulator.generate_geometric_noise((3, 3), 0.0))

    def test_fixed_maps_are_restored_after_temporary_frame_resize(self):
        """Проверяет возврат карты при восстановлении прежнего размера кадра.

        Для двух ненулевых фиксированных шумов размер B создаёт отдельные карты,
        а возврат к A использует старые T#0/G#0 и воспроизводит исходный кадр.
        """
        simulator = GaussianFrameSimulator(rng=np.random.default_rng(48))
        common = (3.0, 2.0, 0.8, 100.0, 10_000.0, 5.0, 3.0, True, 16, True)
        frame_a = simulator.simulate(8, 6, *common)
        self.assertEqual((simulator.temporal_noise_map_id, simulator.geometric_noise_map_id), (0, 0))
        simulator.simulate(10, 8, *common)
        self.assertEqual((simulator.temporal_noise_map_id, simulator.geometric_noise_map_id), (1, 1))
        restored_a = simulator.simulate(8, 6, *common)
        self.assertEqual((simulator.temporal_noise_map_id, simulator.geometric_noise_map_id), (0, 0))
        np.testing.assert_array_equal(restored_a, frame_a)

    def test_clean_optical_frame_is_reused_when_only_noise_changes(self):
        """Проверяет кэширование наиболее дорогой чистой оптической составляющей.

        При одинаковых shape/x0/y0/sigma/сигнале/фоне новый временной шум меняет
        итоговый кадр, но внутренний массив чистой ФРТ остаётся тем же объектом.
        """
        simulator = GaussianFrameSimulator(rng=np.random.default_rng(49))
        arguments = (64, 48, 31.2, 23.8, 1.1, 1000.0, 20.0)
        first = simulator.simulate(*arguments, 2.0, 0.0, True, 16, False)
        cached_clean = simulator._clean_frame
        second = simulator.simulate(*arguments, 2.0, 0.0, True, 16, False)
        self.assertIs(simulator._clean_frame, cached_clean)
        self.assertFalse(np.array_equal(first, second))

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

    def test_robust_fit_profiles_amplitude_and_background_in_raw_lsb(self):
        """Проверяет совместную оценку x0/y0/sigma/A/B без обрезки данных.

        Фон и амплитуда не передаются как известные параметры модели; рамка
        задаёт лишь мягкую оценку B и масштаб шума для Huber-функционала.
        """
        expected = (1.2, 0.8, 0.6)
        amplitude, background = 5_000.0, 1_000.0
        roi = background + amplitude * model_image((3, 3), *expected)
        result = fit_gaussian(
            roi, FIT_METHOD_ROBUST_NELDER_MEAD,
            background_level=background, subtract_background=True,
            noise_sigma=20.0, background_prior_sigma=5.0,
        )
        self.assertTrue(result["success"])
        self.assertTrue(result["robust"])
        np.testing.assert_allclose(
            [result["x0"], result["y0"], result["sigma"]], expected, atol=2e-5,
        )
        self.assertAlmostEqual(result["A"], amplitude, places=2)
        self.assertAlmostEqual(result["fitted_background"], background, places=2)
        self.assertTrue(result["background_prior_used"])

    def test_huber_fit_rejects_single_bright_outlier_in_three_by_three_roi(self):
        """Сравнивает старую и робастную оценки при одиночном выбросе.

        В левый верхний отсчёт 3×3 добавлена помеха 1500 LSB. Старое
        яркостное взвешивание принимает её за часть ФРТ, а Huber снижает вес.
        """
        expected = (1.2, 0.8, 0.6)
        roi = 1_000.0 + 5_000.0 * model_image((3, 3), *expected)
        roi[0, 0] += 1_500.0
        legacy = fit_gaussian(
            roi, FIT_METHOD_NELDER_MEAD, background_level=1_000.0,
            subtract_background=True, noise_sigma=20.0,
        )
        robust = fit_gaussian(
            roi, FIT_METHOD_ROBUST_NELDER_MEAD, background_level=1_000.0,
            subtract_background=True, noise_sigma=20.0,
            background_prior_sigma=5.0,
        )
        self.assertLess(abs(robust["sigma"] - expected[2]), abs(legacy["sigma"] - expected[2]))
        self.assertLess(abs(robust["sigma"] - expected[2]), 0.01)
        self.assertEqual(robust["outlier_count"], 1)
        self.assertEqual(robust["outlier_coordinates"], [(0, 0)])
        self.assertLess(robust["robust_weights"][0, 0], 0.1)

    def test_quadrant_robust_fit_uses_quadrant_only_as_initialization(self):
        """Проверяет полный метод квадранты → robust Nelder–Mead.

        Итоговый центр является непрерывной оценкой и не зажат выбранным
        квадрантом; трасса дополнительно хранит профилированные A и B.
        """
        expected = (1.7, 0.3, 0.8)
        roi = 700.0 + 4_000.0 * model_image((3, 3), *expected)
        result = fit_gaussian(
            roi, FIT_METHOD_QUADRANT_ROBUST_NELDER_MEAD,
            background_level=700.0, subtract_background=True,
            noise_sigma=10.0, background_prior_sigma=2.0,
        )
        self.assertEqual(result["quadrant"]["selected_quadrants"], ["RT"])
        np.testing.assert_allclose(
            [result["x0"], result["y0"], result["sigma"]], expected, atol=3e-5,
        )
        self.assertIn("amplitude", result["optimization_trace"][0])
        self.assertIn("background", result["optimization_trace"][0])

    def test_robust_fit_has_no_optics_derived_sigma_bounds(self):
        """Проверяет свободную оценку широкой ФРТ без априорного диапазона.

        sigma=3 px восстанавливается из ROI 7×7; в алгоритм не передаются
        параметры объектива, длина волны или допустимые min/max ширины.
        """
        expected = (3.2, 2.8, 3.0)
        roi = 500.0 + 9_000.0 * model_image((7, 7), *expected)
        result = fit_gaussian(
            roi, FIT_METHOD_ROBUST_NELDER_MEAD, background_level=500.0,
            subtract_background=True, noise_sigma=1e-3,
            background_prior_sigma=1e-3,
        )
        self.assertTrue(result["success"])
        np.testing.assert_allclose(
            [result["x0"], result["y0"], result["sigma"]], expected, atol=2e-5,
        )

    def test_background_ring_reports_robust_statistics_under_an_outlier(self):
        """Проверяет устойчивые median/MAD при выбросе в фоновой рамке.

        Среднее и обычное СКО должны измениться, тогда как медиана и robust_std
        сохраняют уровень основной совокупности фоновых пикселей.
        """
        image = np.full((11, 11), 1_000.0)
        image[2, 5] = 50_000.0
        statistics = estimate_background_ring(
            image, 5, 5, roi_size=3, ring_width=2, ring_gap=0,
        )
        self.assertGreater(statistics.mean, statistics.median)
        self.assertGreater(statistics.std, 0.0)
        self.assertEqual(statistics.median, 1_000.0)
        self.assertEqual(statistics.mad, 0.0)
        self.assertEqual(statistics.robust_std, 0.0)


if __name__ == "__main__":
    unittest.main()
