from __future__ import annotations

import numpy as np
import pandas as pd


def safe_sample(df: pd.DataFrame, max_points: int | None) -> pd.DataFrame:
    """
    Sample a DataFrame to a maximum number of points while keeping
    first/last samples. If max_points is None, returns a copy unchanged.
    """
    if max_points is None:
        return df.copy()

    max_points = max(1, int(max_points))
    n = len(df)
    if n <= max_points:
        return df.copy()

    step = max(1, n // max_points)
    idx = np.arange(0, n, step, dtype=int)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return df.iloc[idx].copy()


def apply_time_filter(
    df: pd.DataFrame,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """
    Apply optional time-range filtering.
    - If DateTimeIndex exists, filter by index.
    - Else if a common time column exists, filter by that parsed column.
    - Else return unchanged copy.
    """
    if start is None and end is None:
        return df.copy()

    if isinstance(df.index, pd.DatetimeIndex):
        out = df
        if start is not None:
            out = out.loc[out.index >= pd.Timestamp(start)]
        if end is not None:
            out = out.loc[out.index <= pd.Timestamp(end)]
        return out.copy()

    time_col = None
    for candidate in ("timestamp", "time", "datetime", "date"):
        if candidate in df.columns:
            time_col = candidate
            break

    if time_col is None:
        return df.copy()

    parsed = pd.to_datetime(df[time_col], errors="coerce")
    mask = parsed.notna()
    if start is not None:
        mask &= parsed >= pd.Timestamp(start)
    if end is not None:
        mask &= parsed <= pd.Timestamp(end)
    return df.loc[mask].copy()


def select_active_fractal_levels(
    fractals: dict,
    macd_threshold_ratio: float = 0.001,
    log_prefix: str = "[viz]",
) -> list:
    """
    Return sorted fractal levels with non-degenerate close_F and meaningful c_macd.
    """
    active_levels = []
    for F in sorted(fractals.keys()):
        df_check = fractals[F]

        if "close_F" in df_check.columns and df_check["close_F"].nunique() <= 1:
            print(f"{log_prefix} F={F} übersprungen — close_F hat nur 1 Wert")
            continue

        if "c_macd" in df_check.columns:
            macd_range = df_check["c_macd"].abs().max()
            threshold = float(F) * macd_threshold_ratio
            if pd.isna(macd_range) or macd_range < threshold:
                print(
                    f"{log_prefix} F={F} übersprungen — "
                    f"c_macd max={macd_range:.4f} < threshold={threshold:.4f}"
                )
                continue

        active_levels.append(F)

    return active_levels
