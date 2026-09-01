import unittest

import numpy as np

from gaussian_app import GaussianFrameSimulator
from gaussian_math import (
    crop_around_max,
    fit_gaussian_weighted,
    gaussian_pixel_integral,
    lsb_to_watts,
    model_image,
    watts_to_lsb,
)


class GaussianMathTests(unittest.TestCase):
    def test_vectorized_model_matches_scalar_pixel_integrals(self):
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
        for expected in ((1.0, 1.0, 0.63), (1.25, 0.8, 1.0), (0.65, 1.35, 1.5)):
            with self.subTest(expected=expected):
                result = fit_gaussian_weighted(model_image((3, 3), *expected))
                self.assertTrue(result["success"])
                np.testing.assert_allclose(
                    [result["x0"], result["y0"], result["sigma"]], expected, rtol=0, atol=2e-5
                )

    def test_fixed_geometric_pattern_scales_with_requested_sigma(self):
        simulator = GaussianFrameSimulator(rng=np.random.default_rng(42))
        common = (8, 6, 3.0, 2.0, 0.8, 0.0, 10_000.0, 0.0)
        frame_sigma_2 = simulator.simulate(*common, 2.0, True, 16)
        frame_sigma_5 = simulator.simulate(*common, 5.0, True, 16)
        np.testing.assert_allclose(frame_sigma_5 - 10_000.0, 2.5 * (frame_sigma_2 - 10_000.0))

    def test_crop_at_edge_keeps_requested_shape(self):
        image = np.zeros((4, 4))
        image[0, 0] = 1.0
        crop, x_max, y_max = crop_around_max(image, 3)
        self.assertEqual(crop.shape, (3, 3))
        self.assertEqual((x_max, y_max), (0, 0))

    def test_crop_rejects_even_size(self):
        with self.assertRaises(ValueError):
            crop_around_max(np.ones((3, 3)), 2)

    def test_lsb_watt_conversion_round_trip(self):
        values = np.array([0.0, 1.0, 60.0, 30_000.0])
        np.testing.assert_allclose(watts_to_lsb(lsb_to_watts(values, 60.0), 60.0), values)
        with self.assertRaises(ValueError):
            lsb_to_watts(values, 0.0)


if __name__ == "__main__":
    unittest.main()
