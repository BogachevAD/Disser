"""Анализ экспериментального кадра и вспомогательная модель дальности.

Файл сохранён как отдельный исследовательский сценарий: он читает текстовую
матрицу, оценивает фон по кольцу, подгоняет пиксельно-интегрированную гауссиану
и строит диагностические изображения. Жёстких локальных путей и выполнения при
импорте нет; путь к измерению передаётся через командную строку.
"""

import argparse
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from gaussian_math import crop_around_max, crop_around_pixel, fit_gaussian_weighted, local_to_global


def estimate_background_annulus(image, center_x, center_y, inner_size=3, outer_size=9):
    """Оценивает постоянный фон по квадратному кольцу вокруг цели.

    image — кадр, center_x/center_y — пиксель цели; inner_size исключает сигнал,
    outer_size задаёт внешнюю границу. Возвращается медиана кольца, устойчивая
    к одиночным выбросам и дефектным пикселям.
    """
    if inner_size <= 0 or outer_size <= inner_size or inner_size % 2 == 0 or outer_size % 2 == 0:
        raise ValueError("inner_size и outer_size должны быть нечётными, outer_size > inner_size")
    window, _, _ = crop_around_pixel(image, center_x, center_y, outer_size)
    center = outer_size // 2
    y, x = np.indices(window.shape)
    chebyshev_radius = np.maximum(np.abs(x - center), np.abs(y - center))
    mask = chebyshev_radius > inner_size // 2
    return float(np.median(window[mask]))


def analyze_measurement(file_path, border=5, roi_size=3, background_window=9, show_plots=True):
    """Выполняет полный анализ текстовой матрицы экспериментального кадра.

    file_path указывает файл np.loadtxt; border удаляет краевые строки/столбцы;
    roi_size и background_window задают окно ФРТ и оценки фона. Возвращаемый
    словарь содержит исходные данные, ROI, фон, fit и глобальные координаты.
    """
    file_path = Path(file_path)
    data = np.loadtxt(file_path)
    if data.ndim != 2 or min(data.shape) <= 2 * border:
        raise ValueError("Файл должен содержать двумерную матрицу больше удаляемой границы")

    # Подготовка кадра: удаляем ненадёжную границу и находим кандидат по максимуму.
    cropped = data[border:-border, border:-border] if border else data.copy()
    roi, center_x, center_y = crop_around_max(cropped, roi_size)

    # Радиометрическая коррекция: медианный фон кольца вычитается из ROI.
    background = estimate_background_annulus(
        cropped, center_x, center_y, inner_size=roi_size, outer_size=background_window
    )
    corrected_roi = np.clip(roi - background, 0.0, None)

    # Субпиксельная оценка: локальный fit переводится в координаты cropped и data.
    fit = fit_gaussian_weighted(corrected_roi)
    roi_origin_x = center_x - roi_size // 2
    roi_origin_y = center_y - roi_size // 2
    fitted_x, fitted_y = local_to_global(fit["x0"], fit["y0"], roi_origin_x, roi_origin_y)
    fitted_x_source, fitted_y_source = fitted_x + border, fitted_y + border

    result = {
        "file_path": file_path,
        "data": data,
        "cropped": cropped,
        "roi": roi,
        "corrected_roi": corrected_roi,
        "background": background,
        "center_pixel": (center_x, center_y),
        "fitted_center_cropped": (fitted_x, fitted_y),
        "fitted_center_source": (fitted_x_source, fitted_y_source),
        "fit": fit,
    }
    if show_plots:
        plot_measurement_analysis(result)
    return result


def plot_measurement_analysis(result):
    """Строит ROI измерения и восстановленную модель в общей шкале LSB.

    result передаётся из analyze_measurement(). Первая ось показывает найденный
    центр и окружность sigma, вторая — модель; одинаковые vmin/vmax позволяют
    визуально сравнивать отсчёты без автоматического изменения контраста.
    """
    roi = result["corrected_roi"]
    fit = result["fit"]
    vmin, vmax = float(np.min(roi)), float(np.max(roi))
    if vmax <= vmin:
        vmax = vmin + 1.0

    figure, (roi_axis, model_axis) = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    roi_axis.imshow(roi, cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
    roi_axis.plot(fit["x0"], fit["y0"], "rx", markersize=7, label="Оценённый центр")
    roi_axis.add_patch(
        patches.Circle(
            (fit["x0"], fit["y0"]), fit["sigma"], edgecolor="red",
            facecolor="none", linestyle="--", linewidth=1.2, label="σ",
        )
    )
    roi_axis.set_title("Экспериментальный ROI, фон вычтен")
    roi_axis.legend()
    model_axis.imshow(fit["model_signal"], cmap="gray", interpolation="nearest", vmin=vmin, vmax=vmax)
    model_axis.set_title("Пиксельно-интегрированная модель")
    plt.show()


def E_h2(distance_km, source_parameter):
    """Вычисляет освещённость на входном зрачке по эмпирической модели.

    distance_km — дальность в километрах, source_parameter — агрегированный
    параметр источника DI. Формула сохранена из исходного исследования и требует
    отдельного физического обоснования единиц перед включением в диссертацию.
    """
    distance_km = np.asarray(distance_km, dtype=float)
    inverse_square = source_parameter / ((distance_km * 1e5) ** 2)
    transmission = 10 ** (
        -0.6 - distance_km / 190 + np.exp(-((distance_km + 43) / 47) ** 2.9)
    ) - 0.04
    return inverse_square * transmission


def snr_from_D(distance_km, source_parameter, gain=1.0, noise_sigma=1.0):
    """Переводит освещённость E_h2 в линейное отношение сигнал/шум.

    gain объединяет усиление тракта, noise_sigma — СКО шума в согласованных
    единицах. Отрицательная часть эмпирической освещённости ограничивается нулём.
    """
    if noise_sigma <= 0:
        raise ValueError("noise_sigma должна быть положительной")
    irradiance = np.maximum(E_h2(distance_km, source_parameter), 0.0)
    return gain * irradiance / noise_sigma


def Dmax_from_snr_threshold(
    snr_threshold, source_parameter, gain=1.0, noise_sigma=1.0,
    distance_min=0.1, distance_max=120.0, grid_size=20_000,
):
    """Ищет максимальную дальность, где SNR не ниже заданного порога.

    Параметры задают порог, источник, тракт, шум и равномерную сетку дальности.
    Возвращается последняя допустимая дальность или NaN, если порог недостижим.
    """
    distance_grid = np.linspace(distance_min, distance_max, grid_size)
    snr_grid = snr_from_D(distance_grid, source_parameter, gain, noise_sigma)
    valid = np.flatnonzero(snr_grid >= snr_threshold)
    return np.nan if valid.size == 0 else float(distance_grid[valid[-1]])


def plot_Dmax_vs_SNR(
    source_parameter, gain=1.0, noise_sigma=1.0,
    snr_min_db=-10.0, snr_max_db=30.0, points=120,
):
    """Строит зависимость предельной дальности от требуемого SNR.

    Диапазон snr_min_db…snr_max_db переводится в линейные значения; для каждого
    порога вызывается Dmax_from_snr_threshold(), после чего строится график.
    """
    snr_db = np.linspace(snr_min_db, snr_max_db, points)
    snr_linear = 10 ** (snr_db / 10.0)
    distances = np.array([
        Dmax_from_snr_threshold(value, source_parameter, gain, noise_sigma)
        for value in snr_linear
    ])
    plt.figure(figsize=(8, 5))
    plt.plot(snr_db, distances, linewidth=2)
    plt.xlabel("Требуемый порог SNR, дБ")
    plt.ylabel("Максимальная дальность, км")
    plt.title("Dmax(SNR) по эмпирической модели E_h2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def build_argument_parser():
    """Создаёт CLI-парсер анализа экспериментального файла.

    Входных переменных нет; возвращается ArgumentParser с четырьмя параметрами.
    """
    parser = argparse.ArgumentParser(description="Оценка гауссовой ФРТ по текстовой матрице")
    parser.add_argument("file", type=Path, help="Путь к TXT-файлу, читаемому numpy.loadtxt")
    parser.add_argument("--border", type=int, default=5, help="Число удаляемых краевых пикселей")
    parser.add_argument("--roi-size", type=int, default=3, choices=(3, 5, 7), help="Размер ROI")
    parser.add_argument("--background-window", type=int, default=9, help="Внешний размер фонового кольца")
    return parser


def main(argv=None):
    """Разбирает argv, запускает анализ и печатает оценки.

    argv — необязательный список CLI-аргументов; None использует sys.argv.
    """
    arguments = build_argument_parser().parse_args(argv)
    result = analyze_measurement(
        arguments.file, arguments.border, arguments.roi_size, arguments.background_window, True
    )
    fit = result["fit"]
    print(f"Фон: {result['background']:.6g} LSB")
    print(f"Центр в исходной матрице: ({result['fitted_center_source'][0]:.6f}, "
          f"{result['fitted_center_source'][1]:.6f})")
    print(f"Sigma: {fit['sigma']:.6f} px; success={fit['success']}; loss={fit['loss']:.6e}")


if __name__ == "__main__":
    # Прямой запуск требует путь к измерению; импорт функций ничего не вычисляет.
    main()
