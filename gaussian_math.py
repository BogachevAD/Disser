"""Математические функции для моделирования и оценки гауссова пятна."""

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr


def lsb_to_watts(value_lsb, lsb_per_picowatt):
    """Переводит разряды АЦП в Ватты."""
    if lsb_per_picowatt <= 0:
        raise ValueError("lsb_per_picowatt должен быть положительным")
    return np.asarray(value_lsb, dtype=float) / lsb_per_picowatt * 1e-12


def watts_to_lsb(value_watts, lsb_per_picowatt):
    """Переводит Ватты в разряды АЦП."""
    if lsb_per_picowatt <= 0:
        raise ValueError("lsb_per_picowatt должен быть положительным")
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

    if sigma <= 0:
        raise ValueError("sigma должна быть положительной")

    def phi(t):
        return ndtr(t / sigma)

    x1, x2 = i - 0.5, i + 0.5
    y1, y2 = j - 0.5, j + 0.5
    return (phi(x2 - x0) - phi(x1 - x0)) * (phi(y2 - y0) - phi(y1 - y0))


def model_image(shape, x0, y0, sigma):
    """Возвращает нормированную на сумму 1 гауссову модель размера (height, width)."""
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("Размеры изображения должны быть положительными")
    if sigma <= 0:
        raise ValueError("sigma должна быть положительной")

    # Интеграл круговой 2D-гауссианы сепарабелен. Внешнее произведение
    # одномерных интегралов экономит четыре полноразмерных временных массива.
    x = np.arange(width, dtype=float)
    y = np.arange(height, dtype=float)
    x_mass = ndtr((x + 0.5 - x0) / sigma) - ndtr((x - 0.5 - x0) / sigma)
    y_mass = ndtr((y + 0.5 - y0) / sigma) - ndtr((y - 0.5 - y0) / sigma)
    image = np.outer(y_mass, x_mass)
    total = np.sum(image)
    if total <= 0:
        return image
    return image / total


def fit_gaussian_weighted(pixels):
    """Подгоняет x0, y0 и sigma к 2D-матрице взвешенным МНК.

    Координаты центра и sigma ограничены физически допустимой областью. Это
    предотвращает уход оптимизатора в отрицательную ширину или далеко за ROI.
    """
    pixels = np.asarray(pixels, dtype=float)
    if pixels.ndim != 2 or pixels.size == 0:
        raise ValueError("pixels должна быть непустой двумерной матрицей")
    if not np.all(np.isfinite(pixels)):
        raise ValueError("pixels содержит NaN или бесконечные значения")

    pixels = normalize_signal_sum1(pixels)
    height, width = pixels.shape
    weights = np.clip(pixels, 0.0, None)

    if np.sum(weights) <= 0:
        x0 = (width - 1.0) / 2.0
        y0 = (height - 1.0) / 2.0
        return {
            "x0": x0,
            "y0": y0,
            "sigma": 1.0,
            "A": 0.0,
            "success": False,
            "loss": 0.0,
            "model": np.zeros_like(pixels),
            "abs_errors": np.zeros_like(pixels),
            "weights": weights,
            "message": "В ROI отсутствует положительный сигнал",
        }

    x_grid = np.arange(width, dtype=float)
    y_grid = np.arange(height, dtype=float)
    x0_init = float(np.sum(weights * x_grid[None, :]))
    y0_init = float(np.sum(weights * y_grid[:, None]))
    radial_variance = np.sum(
        weights
        * ((x_grid[None, :] - x0_init) ** 2 + (y_grid[:, None] - y0_init) ** 2)
    ) / 2.0
    # Дисперсия интегрированного по площади пикселя сигнала включает 1/12 px².
    sigma_init = float(np.sqrt(max(radial_variance - 1.0 / 12.0, 0.05**2)))
    sqrt_weights = np.sqrt(weights)

    def residuals(params):
        x0, y0, sigma = params
        model = model_image((height, width), x0, y0, sigma)
        return (sqrt_weights * (pixels - model)).ravel()

    result = least_squares(
        residuals,
        [x0_init, y0_init, sigma_init],
        bounds=([-0.5, -0.5, 0.05], [width - 0.5, height - 0.5, 2.0 * max(height, width)]),
        method="trf",
    )
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
        "loss": float(np.sum(result.fun**2)),
        "model": fitted_model,
        "abs_errors": np.abs(pixels - fitted_model),
        "weights": weights,
        "message": result.message,
    }


def crop_around_max(image, size=3):
    """Вырезает окно size×size вокруг максимального пикселя с дополнением краев."""
    image = np.asarray(image)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("image должна быть непустой двумерной матрицей")
    if not isinstance(size, (int, np.integer)) or size <= 0 or size % 2 == 0:
        raise ValueError("size должен быть положительным нечетным целым числом")
    half = size // 2
    y_max, x_max = np.unravel_index(np.argmax(image), image.shape)
    padded = np.pad(image, half, mode="edge")
    y_pad, x_pad = y_max + half, x_max + half
    crop = padded[y_pad - half : y_pad + half + 1, x_pad - half : x_pad + half + 1]
    return crop, x_max, y_max
