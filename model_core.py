import json
from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import pandas as pd


def mean_relative_error_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Средняя относительная погрешность, %.

    Расчет:
        δср = mean(abs((Ycalc - Yexp) / Yexp)) * 100
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    denom = np.maximum(np.abs(y_true), 1e-9)
    return float(np.mean(np.abs((y_pred - y_true) / denom)) * 100.0)


def relative_error_percent(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Относительная погрешность по каждой точке, %.
    Возвращается модуль ошибки.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    denom = np.maximum(np.abs(y_true), 1e-9)
    return np.abs((y_pred - y_true) / denom) * 100.0


def signed_relative_error_percent(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Знаковая относительная погрешность по каждой точке, %.
    Используется для графика, чтобы было видно направление отклонения.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

    denom = np.maximum(np.abs(y_true), 1e-9)
    return (y_pred - y_true) / denom * 100.0


def mape_percent(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Оставлено для совместимости со старой версией кода.
    В интерфейсе выводится не как MAPE, а как средняя относительная погрешность δср, %.
    """
    return mean_relative_error_percent(y_true, y_pred)


@dataclass
class LinearModelSpec:
    """
    Расчетная модель одного выходного параметра:

        Y = K0 + k1*X1 + k2*X2 + ... + kn*Xn

    K0 — свободный коэффициент уравнения, полученный в Approx.
    В Simulink каждый коэффициент k дополнительно представлен
    передаточной функцией k/(T*s + 1), чтобы учесть инерционность процесса.
    """

    y_name: str
    title: str
    x_vars: List[str]
    coef: List[float]
    k0: float
    tau: float
    unit: str = ""

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        missing = [x for x in self.x_vars if x not in df.columns]

        if missing:
            raise ValueError(
                f"Для модели {self.y_name} отсутствуют столбцы: {missing}"
            )

        X = df[self.x_vars].to_numpy(dtype=float)
        k = np.asarray(self.coef, dtype=float)

        if X.shape[1] != len(k):
            raise ValueError(
                f"Для {self.y_name}: число входов {X.shape[1]} "
                f"не совпадает с числом коэффициентов {len(k)}"
            )

        return self.k0 + X @ k

    def adapt_k0(self, y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
        """
        Корректировка свободного коэффициента K0:

            K0_new = K0_old + alpha * mean(Yexp - Ycalc)

        Такой вариант адаптации не изменяет коэффициенты при технологических входах,
        а корректирует постоянное смещение модели.
        """
        y_true = np.asarray(y_true, dtype=float).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)

        correction = float(np.mean(y_true - y_pred))
        self.k0 = float(self.k0 + alpha * correction)

        return self.k0


class HVP2DigitalModel:
    """Много-выходная расчетная модель ХВП-2."""

    def __init__(self, models: List[LinearModelSpec]):
        self.models = models
        self.by_name = {m.y_name: m for m in models}

    def get_model(self, y_name: str) -> LinearModelSpec:
        if y_name not in self.by_name:
            raise KeyError(f"Модель {y_name} не найдена")
        return self.by_name[y_name]

    def predict_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Расчет всех выходов Y1...Y18.

        Y18 зависит от расчетных Y8, Y15, Y16, Y17,
        поэтому модели рассчитываются последовательно.
        """
        work = df.copy()
        result = pd.DataFrame(index=df.index)

        for model in self.models:
            y_calc = model.predict(work)
            work[model.y_name] = y_calc
            result[model.y_name] = y_calc

        return result

    def predict_one(self, df: pd.DataFrame, y_name: str) -> np.ndarray:
        """
        Расчет одного выхода.

        Если выбран Y18, предварительно рассчитываются выходы,
        от которых зависит интегральный показатель качества.
        """
        if y_name == "Y18_Qualityindex":
            all_y = self.predict_all(df)
            return all_y[y_name].to_numpy(dtype=float)

        model = self.get_model(y_name)
        return model.predict(df)

    def evaluate_one(
        self,
        df: pd.DataFrame,
        y_name: str,
        y_true_col: str
    ) -> Dict:
        model = self.get_model(y_name)

        y_true = df[y_true_col].to_numpy(dtype=float)
        y_pred = self.predict_one(df, y_name)

        return {
            "Y": y_name,
            "title": model.title,
            "unit": model.unit,
            "tau": model.tau,
            "k0": model.k0,
            "n_rows": int(len(df)),
            "mean_error_percent": mean_relative_error_percent(y_true, y_pred),
            "max_error_percent": float(np.max(relative_error_percent(y_true, y_pred))),
            "mean_Y_exp": float(np.mean(y_true)),
            "mean_Y_calc": float(np.mean(y_pred)),
        }

    def adapt_one(
        self,
        df: pd.DataFrame,
        y_name: str,
        y_true_col: str,
        alpha: float = 0.2,
        threshold_percent: float = 5.0
    ) -> Dict:
        model = self.get_model(y_name)

        y_true = df[y_true_col].to_numpy(dtype=float)
        y_pred_before = self.predict_one(df, y_name)

        k0_before = model.k0
        mean_error_before = mean_relative_error_percent(y_true, y_pred_before)

        adapted = False

        if mean_error_before > threshold_percent:
            model.adapt_k0(y_true, y_pred_before, alpha=alpha)
            adapted = True

        y_pred_after = self.predict_one(df, y_name)
        mean_error_after = mean_relative_error_percent(y_true, y_pred_after)

        return {
            "Y": y_name,
            "title": model.title,
            "unit": model.unit,
            "tau": model.tau,
            "k0_before": k0_before,
            "k0_after": model.k0,
            "mean_error_before_percent": mean_error_before,
            "mean_error_after_percent": mean_error_after,
            "adapted": adapted,
            "threshold_percent": threshold_percent,
            "alpha": alpha,
        }

    def to_json_bytes(self) -> bytes:
        data = {"models": [asdict(m) for m in self.models]}
        return json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")


INPUT_INFO = {
    "X1_Qfeed": "Расход исходной воды",
    "X2_Pfeed": "Давление исходной воды",
    "X3_TafterHX": "Температура после теплообменника",
    "X4_TurbUFin": "Мутность перед ультрафильтрацией",
    "X5_CondUOO2in": "Электропроводность на входе УОО",
    "X6_DoseAS": "Дозирование антискаланта",
    "X7_DoseMBS": "Дозирование MBS",
    "X8_DoseNaOCl": "Дозирование NaOCl",
    "X9_DoseNaOH": "Дозирование NaOH",
    "X10_DoseHCl": "Дозирование HCl",
    "X11_UFwashpressure": "Давление промывки УФ",
    "X12_ROHPspeed": "Скорость насосов УОО",
}


INPUT_LIMITS = {
    "X1_Qfeed": {
        "min": 0.0,
        "max": 500.0,
        "unit": "м3/ч",
        "name": "Расход исходной воды",
    },
    "X2_Pfeed": {
        "min": 0.0,
        "max": 1.6,
        "unit": "МПа",
        "name": "Давление исходной воды",
    },
    "X3_TafterHX": {
        "min": 0.0,
        "max": 60.0,
        "unit": "°C",
        "name": "Температура после теплообменника",
    },
    "X4_TurbUFin": {
        "min": 0.0,
        "max": 100.0,
        "unit": "NTU",
        "name": "Мутность перед ультрафильтрацией",
    },
    "X5_CondUOO2in": {
        "min": 0.0,
        "max": 100.0,
        "unit": "мкСм/см",
        "name": "Электропроводность на входе УОО",
    },
    "X6_DoseAS": {
        "min": 0.0,
        "max": 100.0,
        "unit": "%",
        "name": "Дозирование антискаланта",
    },
    "X7_DoseMBS": {
        "min": 0.0,
        "max": 100.0,
        "unit": "%",
        "name": "Дозирование MBS",
    },
    "X8_DoseNaOCl": {
        "min": 0.0,
        "max": 100.0,
        "unit": "%",
        "name": "Дозирование NaOCl",
    },
    "X9_DoseNaOH": {
        "min": 0.0,
        "max": 100.0,
        "unit": "%",
        "name": "Дозирование NaOH",
    },
    "X10_DoseHCl": {
        "min": 0.0,
        "max": 100.0,
        "unit": "%",
        "name": "Дозирование HCl",
    },
    "X11_UFwashpressure": {
        "min": 0.0,
        "max": 1.0,
        "unit": "МПа",
        "name": "Давление промывки УФ",
    },
    "X12_ROHPspeed": {
        "min": 0.0,
        "max": 100.0,
        "unit": "%",
        "name": "Скорость насосов УОО",
    },
}


def create_default_hvp2_model() -> HVP2DigitalModel:
    """
    Актуальная структура моделей Y1...Y18.
    Коэффициенты соответствуют расчетной части и схемам Simulink.
    """

    models = [
        LinearModelSpec(
            y_name="Y1_dPstrainer",
            title="Перепад давления на фильтре-ловушке",
            x_vars=["X1_Qfeed", "X2_Pfeed", "X4_TurbUFin"],
            coef=[0.0008448, 0.3504618424945, 0.01084691183008],
            k0=-0.278050903768218,
            tau=15,
            unit="MPa",
        ),
        LinearModelSpec(
            y_name="Y2_dPdisk",
            title="Перепад давления на дисковом фильтре",
            x_vars=["X1_Qfeed", "X2_Pfeed", "X4_TurbUFin"],
            coef=[0.0071372, 0.31958, 0.00966531],
            k0=-0.241168,
            tau=15,
            unit="MPa",
        ),
        LinearModelSpec(
            y_name="Y3_dPUF91",
            title="Перепад давления УФ-9.1",
            x_vars=["X1_Qfeed", "X4_TurbUFin", "X8_DoseNaOCl", "X11_UFwashpressure"],
            coef=[0.00075934, 0.010299, -0.00269241, -0.52987],
            k0=0.258652,
            tau=45,
            unit="MPa",
        ),
        LinearModelSpec(
            y_name="Y4_dPUF92",
            title="Перепад давления УФ-9.2",
            x_vars=["X1_Qfeed", "X4_TurbUFin", "X8_DoseNaOCl", "X11_UFwashpressure"],
            coef=[0.00078108, 0.010299, -0.00269241, -0.529878],
            k0=0.258652,
            tau=45,
            unit="MPa",
        ),
        LinearModelSpec(
            y_name="Y5_dPUF93",
            title="Перепад давления УФ-9.3",
            x_vars=["X1_Qfeed", "X4_TurbUFin", "X8_DoseNaOCl", "X11_UFwashpressure"],
            coef=[0.00073835, 0.00960396, -0.00250275, -0.522187],
            k0=0.253238,
            tau=45,
            unit="MPa",
        ),
        LinearModelSpec(
            y_name="Y6_dPUF94",
            title="Перепад давления УФ-9.4",
            x_vars=["X1_Qfeed", "X4_TurbUFin", "X8_DoseNaOCl", "X11_UFwashpressure"],
            coef=[0.00076267, 0.010375, -0.00270522, -0.51899],
            k0=0.264271,
            tau=45,
            unit="MPa",
        ),
        LinearModelSpec(
            y_name="Y7_dPUF95",
            title="Перепад давления УФ-9.5",
            x_vars=["X1_Qfeed", "X4_TurbUFin", "X8_DoseNaOCl", "X11_UFwashpressure"],
            coef=[0.00073668, 0.010416, -0.00249336, -0.527319],
            k0=0.253238,
            tau=45,
            unit="MPa",
        ),
        LinearModelSpec(
            y_name="Y8_TurbUFout",
            title="Мутность после ультрафильтрации",
            x_vars=["X1_Qfeed", "X4_TurbUFin", "X8_DoseNaOCl", "X11_UFwashpressure"],
            coef=[0.00073453, 0.011074, -0.0026184, -0.520768],
            k0=0.234476,
            tau=45,
            unit="NTU",
        ),
        LinearModelSpec(
            y_name="Y9_CondRO17perm",
            title="Электропроводность пермеата УОО-17",
            x_vars=["X5_CondUOO2in", "X12_ROHPspeed", "X6_DoseAS", "X7_DoseMBS"],
            coef=[0.140129, 0.011074, -0.063005, -0.0826],
            k0=14.484512,
            tau=90,
            unit="uS/cm",
        ),
        LinearModelSpec(
            y_name="Y10_CondRO23perm",
            title="Электропроводность пермеата УОО-23",
            x_vars=["X5_CondUOO2in", "X12_ROHPspeed", "X6_DoseAS", "X7_DoseMBS"],
            coef=[0.12776, -0.05923, -0.063005, -0.075389],
            k0=14.484512,
            tau=90,
            unit="uS/cm",
        ),
        LinearModelSpec(
            y_name="Y11_CondRO24perm",
            title="Электропроводность пермеата УОО-24",
            x_vars=["X5_CondUOO2in", "X12_ROHPspeed", "X6_DoseAS", "X7_DoseMBS"],
            coef=[0.134021, -0.060496, -0.061742, -0.078375],
            k0=13.945176,
            tau=90,
            unit="uS/cm",
        ),
        LinearModelSpec(
            y_name="Y12_dPRO17",
            title="Перепад давления УОО-17",
            x_vars=["X2_Pfeed", "X4_TurbUFin", "X5_CondUOO2in", "X12_ROHPspeed"],
            coef=[0.50831, 0.005974, -0.0029369, 0.0047203],
            k0=-0.364898,
            tau=60,
            unit="MPa",
        ),
        LinearModelSpec(
            y_name="Y13_dPRO23",
            title="Перепад давления УОО-23",
            x_vars=["X2_Pfeed", "X4_TurbUFin", "X5_CondUOO2in", "X12_ROHPspeed"],
            coef=[0.512839, 0.00663628, -0.00315394, 0.00461773],
            k0=-0.372626,
            tau=60,
            unit="MPa",
        ),
        LinearModelSpec(
            y_name="Y14_dPRO24",
            title="Перепад давления УОО-24",
            x_vars=["X2_Pfeed", "X4_TurbUFin", "X5_CondUOO2in", "X12_ROHPspeed"],
            coef=[0.49232, 0.00556943, -0.00289616, 0.00404244],
            k0=-0.321015,
            tau=60,
            unit="MPa",
        ),
        LinearModelSpec(
            y_name="Y15_Recoverytotal",
            title="Коэффициент восстановления",
            x_vars=["X1_Qfeed", "X2_Pfeed", "X5_CondUOO2in", "X12_ROHPspeed"],
            coef=[0.064032, 32.29287, -0.424509, 0.338466],
            k0=33.435501,
            tau=80,
            unit="%",
        ),
        LinearModelSpec(
            y_name="Y16_Condpermfinal",
            title="Итоговая электропроводность пермеата",
            x_vars=["X5_CondUOO2in", "X12_ROHPspeed", "X6_DoseAS", "X7_DoseMBS"],
            coef=[0.13579, -0.061724, -0.060057, -0.07815],
            k0=13.87695,
            tau=90,
            unit="uS/cm",
        ),
        LinearModelSpec(
            y_name="Y17_pHfinal",
            title="Итоговый pH",
            x_vars=["X9_DoseNaOH", "X10_DoseHCl", "X3_TafterHX"],
            coef=[-0.00599095, 0.033443, -0.026975],
            k0=7.908302,
            tau=30,
            unit="pH",
        ),
        LinearModelSpec(
            y_name="Y18_Qualityindex",
            title="Интегральный показатель качества воды",
            x_vars=["Y8_TurbUFout", "Y15_Recoverytotal", "Y16_Condpermfinal", "Y17_pHfinal"],
            coef=[-65.075602, 0.736778, -2.605675, 8.414998],
            k0=-8.883059,
            tau=10,
            unit="%",
        ),
    ]

    return HVP2DigitalModel(models)
