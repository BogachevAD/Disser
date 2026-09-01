"""Точка запуска PyCharm и единый набор начальных параметров модели.

Каждый блок ниже задаёт физически связанную группу переменных. Словарь CONFIG
передаётся в gaussian_app.run_gaussian_simulator(), где значения становятся
начальными состояниями элементов интерфейса.

Запуск:
    python run_gauss_simulator.py
"""

from gaussian_app import run_gaussian_simulator

# === Размер кадра и положение пятна ===
# WIDTH/HEIGHT — число пикселей; X0/Y0 — координаты центра в пикселях.
# SIGMA_PX — СКО круговой гауссовой ФРТ относительно шага матрицы.
WIDTH = 32
HEIGHT = 32
X0 = 16.0
Y0 = 16.0
SIGMA_PX = 1.0

# === Яркость в разрядах АЦП (LSB) ===
# ADC_BITS определяет диапазон [0, 2^bits-1]; MAX_ADC_CODE — насыщение.
# AMPLITUDE_LSB задаёт пик пятна над постоянным BACKGROUND_LSB.
ADC_BITS = 16
MAX_ADC_CODE = 2**ADC_BITS - 1
# AMPLITUDE_LSB — пиковая добавка гауссоиды над фоном.
# BACKGROUND_LSB — постоянный уровень фона каждого пикселя.
AMPLITUDE_LSB = 30000.0
BACKGROUND_LSB = 0.0

# === Шумы в разрядах АЦП (LSB, стандартное отклонение) ===
# TEMPORAL_NOISE_LSB меняется между кадрами; GEOMETRIC_NOISE_LSB задаёт
# пространственную неоднородность. FIX_GEOMETRIC_NOISE сохраняет её рисунок.
TEMPORAL_NOISE_LSB = 0.0
GEOMETRIC_NOISE_LSB = 0.0
FIX_GEOMETRIC_NOISE = True

# === Пересчет единиц ===
# По умолчанию: 60 младших разрядов = 1 пикоВатт.
LSB_PER_PICOWATT = 60.0

# === Формирование области интереса ===
# truth фиксирует ROI по известным X0/Y0 для проверки модели; matched_filter
# сначала обнаруживает пятно по кадру. ROI_SIZE выбирается из 3, 5 или 7.
ROI_MODE = "truth"
ROI_SIZE = 3

# === Метод оценки и статистика фона ===
# FIT_METHOD выбирает алгоритм из выпадающего списка; пока зарегистрирован
# Нелдер–Мид. BACKGROUND_RING_GAP отделяет хвосты ФРТ от фоновой рамки,
# BACKGROUND_RING_WIDTH задаёт её толщину. Средний фон можно вычитать, а его
# СКО — использовать для SNR и chi-square.
FIT_METHOD = "nelder_mead"
BACKGROUND_RING_WIDTH = 1
BACKGROUND_RING_GAP = 3
SUBTRACT_RING_BACKGROUND = True
USE_RING_NOISE = True

CONFIG = {
    "WIDTH": WIDTH,
    "HEIGHT": HEIGHT,
    "X0": X0,
    "Y0": Y0,
    "SIGMA_PX": SIGMA_PX,
    "ADC_BITS": ADC_BITS,
    "MAX_ADC_CODE": MAX_ADC_CODE,
    "AMPLITUDE_LSB": AMPLITUDE_LSB,
    "BACKGROUND_LSB": BACKGROUND_LSB,
    "TEMPORAL_NOISE_LSB": TEMPORAL_NOISE_LSB,
    "GEOMETRIC_NOISE_LSB": GEOMETRIC_NOISE_LSB,
    "FIX_GEOMETRIC_NOISE": FIX_GEOMETRIC_NOISE,
    "LSB_PER_PICOWATT": LSB_PER_PICOWATT,
    "ROI_MODE": ROI_MODE,
    "ROI_SIZE": ROI_SIZE,
    "FIT_METHOD": FIT_METHOD,
    "BACKGROUND_RING_WIDTH": BACKGROUND_RING_WIDTH,
    "BACKGROUND_RING_GAP": BACKGROUND_RING_GAP,
    "SUBTRACT_RING_BACKGROUND": SUBTRACT_RING_BACKGROUND,
    "USE_RING_NOISE": USE_RING_NOISE,
}


if __name__ == "__main__":
    # Этот блок выполняется только при прямом запуске файла, а не при импорте.
    run_gaussian_simulator(CONFIG)
