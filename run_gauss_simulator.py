"""Файл запуска PyCharm: здесь собраны основные константы модели.

Запуск:
    python run_gaussian_simulator.py
"""

from gaussian_app import run_gaussian_simulator

# === Размер кадра и положение пятна ===
WIDTH = 32
HEIGHT = 32
X0 = 16.0
Y0 = 16.0
SIGMA_PX = 1.0

# === Яркость в разрядах АЦП (LSB) ===
ADC_BITS = 16
MAX_ADC_CODE = 2**ADC_BITS - 1
AMPLITUDE_LSB = 30000.0
BACKGROUND_LSB = 1000.0

# === Шумы в разрядах АЦП (LSB, стандартное отклонение) ===
TEMPORAL_NOISE_LSB = 50.0
GEOMETRIC_NOISE_LSB = 20.0
FIX_GEOMETRIC_NOISE = True
GEOMETRIC_SEED = 42

# === Пересчет единиц ===
# По умолчанию: 60 младших разрядов = 1 пикоВатт.
LSB_PER_PICOWATT = 60.0

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
    "GEOMETRIC_SEED": GEOMETRIC_SEED,
    "LSB_PER_PICOWATT": LSB_PER_PICOWATT,
}


if __name__ == "__main__":
    run_gaussian_simulator(CONFIG)