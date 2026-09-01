"""Математическое ядро модели пиксельно-интегрированного гауссова пятна.

Модуль переводит единицы, формирует ФРТ на дискретной матрице, выбирает ROI
в режиме верификации или обнаружения и оценивает субпиксельный центр с sigma.
Центры пикселей имеют целые координаты; границы пикселя равны x±0.5, y±0.5.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.signal import fftconvolve
from scipy.special import ndtr


ROI_MODE_TRUTH = "truth"
ROI_MODE_MATCHED_FILTER = "matched_filter"
FIT_METHOD_NELDER_MEAD = "nelder_mead"


@dataclass(frozen=True)
class RoiSelection:
    """Результат выбора локального окна на полном кадре.

    roi содержит матрицу size×size; center_x/center_y — глобальный пиксель,
    вокруг которого взято окно; origin_x/origin_y — глобальные координаты
    верхнего левого элемента ROI. response хранит карту критерия обнаружения.
    """

    roi: np.ndarray
    center_x: int
    center_y: int
    origin_x: int
    origin_y: int
    mode: str
    response: np.ndarray | None = None


@dataclass(frozen=True)
class BackgroundStatistics:
    """Статистика пиксельной рамки вокруг выбранного ROI.

    mean/median — оценки уровня фона в LSB, std — выборочное СКО флуктуаций,
    pixel_count — число реально использованных пикселей без дополнения границ.
    """

    mean: float
    median: float
    std: float
    pixel_count: int


def _as_valid_image(image, name="image"):
    """Преобразует image в float-массив и проверяет размерность и числа.

    Вход: произвольный массив изображения. Выход: двумерный ndarray без
    NaN/Inf; при нарушении контракта функция возбуждает ValueError.
    """
    array = np.asarray(image, dtype=float)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name} должна быть непустой двумерной матрицей")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} содержит NaN или бесконечные значения")
    return array


def _validate_odd_size(size):
    """Проверяет размер квадратного ROI или ядра.

    size передаётся как целое число пикселей и должен быть положительным и
    нечётным, чтобы у окна существовал единственный центральный пиксель.
    """
    if not isinstance(size, (int, np.integer)) or size <= 0 or size % 2 == 0:
        raise ValueError("size должен быть положительным нечетным целым числом")


def lsb_to_watts(value_lsb, lsb_per_picowatt):
    """Переводит код АЦП value_lsb в оптическую мощность, Вт.

    lsb_per_picowatt задаёт число младших разрядов на 1 пВт; функция работает
    как со скаляром, так и с массивом.
    """
    if lsb_per_picowatt <= 0:
        raise ValueError("lsb_per_picowatt должен быть положительным")
    return np.asarray(value_lsb, dtype=float) / lsb_per_picowatt * 1e-12


def watts_to_lsb(value_watts, lsb_per_picowatt):
    """Переводит оптическую мощность value_watts в код АЦП, LSB.

    lsb_per_picowatt задаёт коэффициент тракта; операция является обратной
    lsb_to_watts() и поддерживает скаляры и массивы.
    """
    if lsb_per_picowatt <= 0:
        raise ValueError("lsb_per_picowatt должен быть положительным")
    return np.asarray(value_watts, dtype=float) / 1e-12 * lsb_per_picowatt


def normalize_pixels_sum1(pixels):
    """Сдвигает матрицу к нулю и нормирует её сумму к единице.

    pixels — измеренный ROI. Метод оставлен для совместимости со старым
    анализом, но вычитание минимума искажает ненулевые хвосты гауссианы.
    """
    pixels = _as_valid_image(pixels, "pixels")
    pixels = pixels - np.min(pixels)
    maximum = np.max(pixels)
    if maximum > 0:
        pixels = pixels / maximum
    total = np.sum(pixels)
    return pixels / total if total > 0 else pixels


def normalize_signal_sum1(pixels):
    """Нормирует неотрицательную составляющую сигнала по сумме.

    pixels — ROI после вычитания фона; отрицательные шумовые отсчёты обнуляются.
    Результат описывает относительное распределение энергии между пикселями.
    """
    pixels = _as_valid_image(pixels, "pixels")
    pixels = np.clip(pixels, 0.0, None)
    total = np.sum(pixels)
    return pixels / total if total > 0 else pixels


def gaussian_pixel_integral(x0, y0, sigma, i, j):
    """Вычисляет долю энергии гауссианы, попавшую в пиксель (i, j).

    x0/y0 — непрерывный центр, sigma — СКО в пикселях, i/j — целочисленный
    центр пикселя. Интеграл считается аналитически через нормальную ФР.
    """
    if sigma <= 0:
        raise ValueError("sigma должна быть положительной")

    def phi(value):
        """Возвращает нормальную ФР для границы value.

        value — расстояние от центра; sigma берётся из внешней функции.
        """
        return ndtr(value / sigma)

    x1, x2 = i - 0.5, i + 0.5
    y1, y2 = j - 0.5, j + 0.5
    return (phi(x2 - x0) - phi(x1 - x0)) * (phi(y2 - y0) - phi(y1 - y0))


def model_image(shape, x0, y0, sigma):
    """Формирует нормированную пиксельно-интегрированную ФРТ.

    shape=(height,width), x0/y0 и sigma заданы в пикселях. Благодаря
    сепарабельности вычисляется внешнее произведение двух профилей; сумма
    возвращаемой матрицы равна единице в пределах заданного кадра.
    """
    height, width = shape
    if height <= 0 or width <= 0:
        raise ValueError("Размеры изображения должны быть положительными")
    if sigma <= 0:
        raise ValueError("sigma должна быть положительной")

    x = np.arange(width, dtype=float)
    y = np.arange(height, dtype=float)
    x_mass = ndtr((x + 0.5 - x0) / sigma) - ndtr((x - 0.5 - x0) / sigma)
    y_mass = ndtr((y + 0.5 - y0) / sigma) - ndtr((y - 0.5 - y0) / sigma)
    image = np.outer(y_mass, x_mass)
    total = np.sum(image)
    return image / total if total > 0 else image


def nearest_pixel_center(coordinate, limit):
    """Преобразует непрерывную координату в индекс ближайшего пикселя.

    При точном попадании на границу n+0.5 выбирается пиксель n+1; limit — число
    пикселей по оси и одновременно используется для ограничения результата.
    """
    if limit <= 0:
        raise ValueError("limit должен быть положительным")
    return int(np.clip(np.floor(float(coordinate) + 0.5), 0, limit - 1))


def crop_around_pixel(image, center_x, center_y, size=3):
    """Вырезает size×size вокруг заданного глобального пикселя.

    center_x/center_y — целочисленные индексы. У границы кадр дополняется
    крайними значениями; вместе с ROI возвращаются координаты его начала.
    """
    image = _as_valid_image(image)
    _validate_odd_size(size)
    height, width = image.shape
    center_x = int(np.clip(center_x, 0, width - 1))
    center_y = int(np.clip(center_y, 0, height - 1))
    half = size // 2
    padded = np.pad(image, half, mode="edge")
    crop = padded[center_y:center_y + size, center_x:center_x + size]
    return crop, center_x - half, center_y - half


def estimate_background_ring(
    image, center_x, center_y, roi_size=3, ring_width=1, ring_gap=0,
):
    """Оценивает средний фон и его СКО по рамке вокруг ROI.

    image — полный кадр, center_x/center_y — центральный пиксель ROI, roi_size —
    его нечётный размер, ring_width — толщина рамки, ring_gap — защитный отступ
    от ROI. Пиксели сигнала и отступа исключаются; у границы используются только
    реальные элементы кадра. Отступ уменьшает попадание хвостов ФРТ в фон.
    """
    image = _as_valid_image(image)
    _validate_odd_size(roi_size)
    if not isinstance(ring_width, (int, np.integer)) or ring_width <= 0:
        raise ValueError("ring_width должен быть положительным целым числом")
    if not isinstance(ring_gap, (int, np.integer)) or ring_gap < 0:
        raise ValueError("ring_gap должен быть неотрицательным целым числом")

    center_x, center_y = int(center_x), int(center_y)
    inner_half = roi_size // 2
    ring_inner_radius = inner_half + int(ring_gap)
    outer_half = ring_inner_radius + int(ring_width)
    y0, y1 = max(0, center_y - outer_half), min(image.shape[0], center_y + outer_half + 1)
    x0, x1 = max(0, center_x - outer_half), min(image.shape[1], center_x + outer_half + 1)
    window = image[y0:y1, x0:x1]
    global_y, global_x = np.indices(window.shape)
    global_y += y0
    global_x += x0
    radius = np.maximum(np.abs(global_x - center_x), np.abs(global_y - center_y))
    values = window[(radius > ring_inner_radius) & (radius <= outer_half)]
    if values.size == 0:
        raise ValueError("Рамка не содержит пикселей")
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return BackgroundStatistics(
        mean=float(np.mean(values)),
        median=float(np.median(values)),
        std=std,
        pixel_count=int(values.size),
    )


def crop_around_position(image, x0, y0, size=3):
    """Формирует ROI по известному непрерывному центру модели.

    x0/y0 переводятся в ближайший пиксель независимо от шума; функция нужна
    для верификации генератора и возвращает RoiSelection с глобальной привязкой.
    """
    image = _as_valid_image(image)
    center_x = nearest_pixel_center(x0, image.shape[1])
    center_y = nearest_pixel_center(y0, image.shape[0])
    roi, origin_x, origin_y = crop_around_pixel(image, center_x, center_y, size)
    return RoiSelection(roi, center_x, center_y, origin_x, origin_y, ROI_MODE_TRUTH)


def matched_filter_response(image, sigma, background=0.0):
    """Вычисляет карту отклика согласованного с ФРТ фильтра.

    image — полный кадр, sigma — ожидаемая ширина пятна, background — известный
    постоянный фон. Корреляция с интегрированной гауссовой ФРТ суммирует энергию
    соседних пикселей и устойчивее одиночного максимума к шуму.
    """
    image = _as_valid_image(image)
    if sigma <= 0:
        raise ValueError("sigma должна быть положительной")
    radius = max(1, int(np.ceil(3.0 * sigma)))
    kernel_size = 2 * radius + 1
    kernel = model_image((kernel_size, kernel_size), radius, radius, sigma)
    return fftconvolve(image - float(background), kernel[::-1, ::-1], mode="same")


def crop_around_detected_target(image, sigma, background=0.0, size=3):
    """Обнаруживает пятно согласованным фильтром и вырезает ROI.

    image, sigma и background задают кадр и ожидаемую ФРТ; size задаёт окно
    последующей оценки. Максимум карты отклика определяет центральный пиксель.
    """
    image = _as_valid_image(image)
    response = matched_filter_response(image, sigma, background)
    center_y, center_x = np.unravel_index(np.argmax(response), response.shape)
    roi, origin_x, origin_y = crop_around_pixel(image, center_x, center_y, size)
    return RoiSelection(
        roi, int(center_x), int(center_y), origin_x, origin_y,
        ROI_MODE_MATCHED_FILTER, response,
    )


def select_roi(image, mode, size, x0, y0, sigma, background=0.0):
    """Выбирает алгоритм формирования ROI для инженерной задачи.

    mode='truth' использует известные x0/y0 и проверяет модель; mode=
    'matched_filter' игнорирует истинный центр и сначала обнаруживает сигнал.
    """
    if mode == ROI_MODE_TRUTH:
        return crop_around_position(image, x0, y0, size)
    if mode == ROI_MODE_MATCHED_FILTER:
        return crop_around_detected_target(image, sigma, background, size)
    raise ValueError(f"Неизвестный режим выбора ROI: {mode}")


def _prepare_fit_signal(pixels, background_level, subtract_background):
    """Подготавливает сигнал ROI для алгоритма оценки.

    pixels содержит исходные LSB; background_level — среднее рамки. При
    subtract_background фон вычитается, после чего отрицательные шумовые
    отклонения обнуляются для совместимости с яркостными весами старого метода.
    """
    pixels = _as_valid_image(pixels, "pixels")
    corrected = pixels - float(background_level) if subtract_background else pixels.copy()
    return np.clip(corrected, 0.0, None)


def fit_gaussian_nelder_mead(
    pixels, background_level=0.0, subtract_background=True, noise_sigma=None,
):
    """Оценивает x0, y0 и sigma взвешенным методом Нелдера–Мида.

    pixels — исходный ROI в LSB; background_level вычитается только при флаге;
    noise_sigma — СКО рамки для SNR и chi-square. Оптимизируются три параметра,
    а яркостные веса и совпадение центральной амплитуды повторяют FRT_main.
    """
    raw_pixels = _as_valid_image(pixels, "pixels")
    fit_signal = _prepare_fit_signal(raw_pixels, background_level, subtract_background)
    total_signal = float(np.sum(fit_signal))
    normalized = normalize_signal_sum1(fit_signal)
    height, width = normalized.shape
    weights = normalized.copy()

    if total_signal <= 0:
        x0 = (width - 1.0) / 2.0
        y0 = (height - 1.0) / 2.0
        zeros = np.zeros_like(normalized)
        return {
            "method": FIT_METHOD_NELDER_MEAD,
            "x0": x0, "y0": y0, "sigma": 1.0, "A": 0.0,
            "success": False, "loss": 0.0, "model": zeros,
            "model_signal": zeros, "fit_signal": fit_signal,
            "abs_errors": zeros, "weights": weights,
            "background_level": float(background_level),
            "background_subtracted": bool(subtract_background),
            "noise_sigma": noise_sigma, "snr_peak": np.nan,
            "chi_square": np.nan, "reduced_chi_square": np.nan,
            "message": "В ROI отсутствует положительный сигнал",
        }

    # Энергетический центроид и второй момент задают стартовый симплекс.
    x_grid = np.arange(width, dtype=float)
    y_grid = np.arange(height, dtype=float)
    x0_init = float(np.sum(weights * x_grid[None, :]))
    y0_init = float(np.sum(weights * y_grid[:, None]))
    radial_variance = np.sum(
        weights * ((x_grid[None, :] - x0_init) ** 2 + (y_grid[:, None] - y0_init) ** 2)
    ) / 2.0
    sigma_init = float(np.sqrt(max(radial_variance - 1.0 / 12.0, 0.05**2)))

    def loss(params):
        """Возвращает яркостно-взвешенную сумму квадратов.

        params=(local_x0,local_y0,sigma); нефизические точки получают штраф.
        """
        local_x0, local_y0, fitted_sigma = params
        if (
            fitted_sigma < 0.05 or fitted_sigma > 2.0 * max(height, width)
            or local_x0 < -0.5 or local_x0 > width - 0.5
            or local_y0 < -0.5 or local_y0 > height - 0.5
        ):
            return 1e6
        model = model_image((height, width), local_x0, local_y0, fitted_sigma)
        return float(np.sum(weights * (normalized - model) ** 2))

    result = minimize(
        loss,
        [x0_init, y0_init, sigma_init],
        method="Nelder-Mead",
        options={"xatol": 1e-9, "fatol": 1e-14, "maxiter": 5000},
    )
    x0, y0, sigma = result.x
    model = model_image((height, width), x0, y0, sigma)
    center_y, center_x = height // 2, width // 2
    central_gauss = model[center_y, center_x]
    amplitude = normalized[center_y, center_x] / central_gauss if central_gauss > 0 else 0.0
    fitted_model = amplitude * model
    model_signal = total_signal * fitted_model

    # СКО рамки не меняет минимум при однородном гауссовом шуме, но задаёт
    # физический масштаб SNR и chi-square для сравнения качества методов.
    valid_noise = noise_sigma is not None and np.isfinite(noise_sigma) and noise_sigma > 0
    signal_peak_above_background = max(float(np.max(raw_pixels)) - float(background_level), 0.0)
    snr_peak = signal_peak_above_background / noise_sigma if valid_noise else np.nan
    chi_square = (
        float(np.sum(((fit_signal - model_signal) / noise_sigma) ** 2))
        if valid_noise else np.nan
    )
    degrees_of_freedom = max(fit_signal.size - 3, 1)
    reduced_chi_square = chi_square / degrees_of_freedom if valid_noise else np.nan
    return {
        "method": FIT_METHOD_NELDER_MEAD,
        "x0": float(x0), "y0": float(y0), "sigma": float(sigma),
        "A": float(amplitude), "success": bool(result.success),
        "loss": float(result.fun), "model": fitted_model,
        "model_signal": model_signal, "fit_signal": fit_signal,
        "abs_errors": np.abs(fit_signal - model_signal),
        "weights": weights, "background_level": float(background_level),
        "background_subtracted": bool(subtract_background),
        "noise_sigma": float(noise_sigma) if valid_noise else None,
        "snr_peak": float(snr_peak), "chi_square": chi_square,
        "reduced_chi_square": reduced_chi_square,
        "message": result.message,
    }


def fit_gaussian(
    pixels, method=FIT_METHOD_NELDER_MEAD, background_level=0.0,
    subtract_background=True, noise_sigma=None,
):
    """Направляет ROI в выбранный алгоритм оценки ФРТ.

    method — строковый идентификатор выпадающего списка; остальные переменные
    передаются методу. Пока зарегистрирован только FIT_METHOD_NELDER_MEAD, а
    новые алгоритмы добавляются сюда без изменения интерфейсной цепочки.
    """
    if method == FIT_METHOD_NELDER_MEAD:
        return fit_gaussian_nelder_mead(
            pixels, background_level, subtract_background, noise_sigma
        )
    raise ValueError(f"Неизвестный метод оценки: {method}")


def fit_gaussian_weighted(pixels):
    """Сохраняет совместимость старого вызова взвешенного fit.

    pixels считается уже подготовленным сигналом без фона; фактический расчёт
    выполняет восстановленный метод Нелдера–Мида.
    """
    return fit_gaussian_nelder_mead(pixels, 0.0, False, None)


def local_to_global(local_x, local_y, origin_x, origin_y):
    """Переводит координаты результата подгонки из ROI в полный кадр.

    local_x/local_y отсчитываются от ROI, origin_x/origin_y задают положение его
    верхнего левого элемента; функция возвращает global_x, global_y.
    """
    return float(origin_x + local_x), float(origin_y + local_y)


def crop_around_max(image, size=3):
    """Совместимый со старым кодом выбор ROI по сырому максимуму.

    Функция возвращает crop, x_max, y_max. Для зашумлённых кадров её применять
    не рекомендуется: используйте crop_around_detected_target().
    """
    image = _as_valid_image(image)
    center_y, center_x = np.unravel_index(np.argmax(image), image.shape)
    crop, _, _ = crop_around_pixel(image, center_x, center_y, size)
    return crop, int(center_x), int(center_y)
