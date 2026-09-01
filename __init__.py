"""Публичный интерфейс модели гауссовой ФРТ и выбора ROI.

Файл позволяет импортировать основные классы и функции из одного места. Он не
запускает GUI и не выполняет расчёты, поэтому пакет безопасно использовать в
тестах, исследовательских сценариях и будущих модулях диссертации.
"""

from gaussian_app import GaussianFrameSimulator, GaussianSimulatorApp, run_gaussian_simulator
from gaussian_math import (
    ROI_MODE_MATCHED_FILTER,
    ROI_MODE_TRUTH,
    RoiSelection,
    crop_around_detected_target,
    crop_around_position,
    fit_gaussian_weighted,
    local_to_global,
    lsb_to_watts,
    matched_filter_response,
    model_image,
    normalize_pixels_sum1,
    normalize_signal_sum1,
    select_roi,
    watts_to_lsb,
)

# __all__ документирует поддерживаемые имена; внутренние helpers остаются скрыты.
__all__ = [
    "GaussianFrameSimulator",
    "GaussianSimulatorApp",
    "ROI_MODE_MATCHED_FILTER",
    "ROI_MODE_TRUTH",
    "RoiSelection",
    "crop_around_detected_target",
    "crop_around_position",
    "fit_gaussian_weighted",
    "local_to_global",
    "lsb_to_watts",
    "matched_filter_response",
    "model_image",
    "normalize_pixels_sum1",
    "normalize_signal_sum1",
    "run_gaussian_simulator",
    "select_roi",
    "watts_to_lsb",
]
