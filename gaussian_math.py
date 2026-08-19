"""Математические функции для моделирования и оценки гауссова пятна."""

import numpy as np
from scipy.optimize import minimize
from scipy.special import erf


def lsb_to_watts(value_lsb, lsb_per_picowatt):
    """Переводит разряды АЦП в Ватты."""
    return np.asarray(value_lsb, dtype=float) / lsb_per_picowatt * 1e-12


def watts_to_lsb(value_watts, lsb_per_picowatt):
    """Переводит Ватты в разряды АЦП."""
    return np.asarray(value_watts, dtype=float) / 1e-12 * lsb_per_picowatt


def normalize_pixels_sum1(pixels):
    """Нормализует матрицу в диапазон [0, 1] с суммой элементов 1.

    Эта функция оставлена для совместимости со старым методом, где из ROI
    дополнительно вычитался минимум. Для проверки идеальной синтетической
    гауссоиды лучше использовать normalize_signal_sum1(), чтобы не искажать
    форму ненулевых хвостов гауссианы.
    """
    pixels = np.asarray(pixels, dtype=float)
    pixels = pixels - np.min(pixels)
    max_val = np.max(pixels)
    if max_val > 0:
        pixels = pixels / max_val
    total = np.sum(pixels)
    if total > 0:
        pixels = pixels / total
    return pixels


def normalize_signal_sum1(pixels):
    """Нормализует неотрицательный сигнал только по сумме без вычитания минимума."""
    pixels = np.clip(np.asarray(pixels, dtype=float), 0.0, None)
    total = np.sum(pixels)
    if total > 0:
        return pixels / total
    return pixels


def gaussian_pixel_integral(x0, y0, sigma, i, j):
    """Интеграл 2D-гауссианы по площади пикселя с центром в (i, j)."""

    def phi(t):
        return 0.5 * (1.0 + erf(t / (np.sqrt(2.0) * sigma)))

    x1, x2 = i - 0.5, i + 0.5
    y1, y2 = j - 0.5, j + 0.5
    return (phi(x2 - x0) - phi(x1 - x0)) * (phi(y2 - y0) - phi(y1 - y0))


def model_image(shape, x0, y0, sigma):
    """Возвращает нормированную на сумму 1 гауссову модель размера (height, width)."""
    height, width = shape
    y_idx, x_idx = np.indices((height, width))
    phi_x2 = 0.5 * (1.0 + erf((x_idx + 0.5 - x0) / (np.sqrt(2.0) * sigma)))
    phi_x1 = 0.5 * (1.0 + erf((x_idx - 0.5 - x0) / (np.sqrt(2.0) * sigma)))
    phi_y2 = 0.5 * (1.0 + erf((y_idx + 0.5 - y0) / (np.sqrt(2.0) * sigma)))
    phi_y1 = 0.5 * (1.0 + erf((y_idx - 0.5 - y0) / (np.sqrt(2.0) * sigma)))
    image = (phi_x2 - phi_x1) * (phi_y2 - phi_y1)
    total = np.sum(image)
    if total <= 0:
        return image
    return image / total


def fit_gaussian_weighted(pixels):
    """Подгоняет x0, y0 и sigma к 2D-матрице тем же взвешенным методом."""
    pixels = np.asarray(pixels, dtype=float)
    height, width = pixels.shape
    weights = np.clip(pixels, 0.0, None)

    def loss(params):
        x0, y0, sigma = params
        if sigma <= 0:
            return np.inf
        model = model_image((height, width), x0, y0, sigma)
        return np.sum(weights * (pixels - model) ** 2)

    result = minimize(loss, [width / 2.0, height / 2.0, 1.0], method="Nelder-Mead")
    x0, y0, sigma = result.x
    model = model_image((height, width), x0, y0, sigma)
    yc, xc = height // 2, width // 2
    central_gauss = model[yc, xc]
    amplitude = pixels[yc, xc] / central_gauss if central_gauss > 0 else 0.0
    fitted_model = amplitude * model
    return {
        "x0": x0,
        "y0": y0,
        "sigma": sigma,
        "A": amplitude,
        "success": result.success,
        "loss": result.fun,
        "model": fitted_model,
        "abs_errors": np.abs(pixels - fitted_model),
        "weights": weights,
    }


def crop_around_max(image, size=3):
    """Вырезает окно size×size вокруг максимального пикселя с дополнением краев."""
    half = size // 2
    y_max, x_max = np.unravel_index(np.argmax(image), image.shape)
    padded = np.pad(image, half, mode="edge")
    y_pad, x_pad = y_max + half, x_max + half
    crop = padded[y_pad - half : y_pad + half + 1, x_pad - half : x_pad + half + 1]
    return crop, x_max, y_max