import numpy as np
import pandas as pd


def build_urdaten(
    series: pd.Series,
    mode: str = "raw",
    scale_factor: float = 1.0,
    center: bool = False,
) -> pd.DataFrame:
    """
    Erzeuge Urdaten aus einer Rohserie.

    Parameter
    ---------
    series : pd.Series
        Eingangsdaten als Zeitreihe.
    mode : str
        "raw"    -> keine Normierung
        "std"    -> Division durch Standardabweichung
        "robust" -> Division durch MAD (Median Absolute Deviation)
    scale_factor : float
        zusätzlicher Faktor zur bewussten Skalierung der Schrittgröße
    center : bool
        falls True, wird der Mittelwert der Schritte vor der Kumulation entfernt
    """
    s = series.dropna().astype(float).copy()

    step_raw = s.diff().fillna(0.0)

    x = step_raw.copy()

    if center:
        x = x - x.mean()

    if mode == "raw":
        step_norm = x

    elif mode == "std":
        std = x.std(ddof=1)
        if pd.notna(std) and std > 0:
            step_norm = x / std
        else:
            step_norm = x.copy()

    elif mode == "robust":
        med = x.median()
        mad = (x - med).abs().median()
        if pd.notna(mad) and mad > 0:
            step_norm = (x - med) / mad
        else:
            step_norm = x.copy()

    else:
        raise ValueError("mode muss 'raw', 'std' oder 'robust' sein")

    step_norm = step_norm * float(scale_factor)

    cum_height = step_norm.cumsum()

    df = pd.DataFrame(
        {
            "value": s,
            "step_raw": step_raw,
            "step_norm": step_norm,
            "cum_height": cum_height,
        },
        index=s.index,
    )

    return df