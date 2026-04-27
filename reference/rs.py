"""
FAIB NEXUS — rs.py
==================
Rescaled Range (R/S) Analyse — klassischer Hurst-Exponent (Hurst 1951)

Berechnet H je Fraktalebene auf den Differenzen von close_F.

H > 0.5 = persistentes Verhalten (Trend hat Gedaechtnis)
H = 0.5 = Zufallsprozess (keine Struktur)
H < 0.5 = antipersistentes Verhalten (Mittelwertrueckkehr)

Aktivierung in main.py:
    ENABLE_RS = True
    import rs
    rs.run_rs(name, ur, fractals, out_dir)
"""

# ============================================================
# PARAMETER — hier anpassen
# ============================================================

# Minimale Fenstergroesse fuer R/S Berechnung
RS_MIN_WINDOW = 8

# Maximale Fenstergroesse (None = automatisch = len/4)
RS_MAX_WINDOW = None

# Anzahl der logarithmisch verteilten Fenstergroessen
RS_N_WINDOWS = 20

# Welche Zeitreihe je Fraktal verwenden:
# "diff_close"  = Differenzen von close_F (empfohlen, stationaer)
# "dot"         = dot-Werte direkt
# "cum_height"  = globale Basisbahn (einmal, nicht je Fraktal)
RS_INPUT = "diff_close"

# Mindestlaenge der Zeitreihe fuer sinnvolle Berechnung
RS_MIN_LENGTH = 100

# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path


def _get_series(df: pd.DataFrame, F: float, mode: str) -> np.ndarray:
    """Extrahiert die Eingangszeitreihe je nach Modus."""
    if mode == "diff_close":
        if "close_F" in df.columns:
            s = df["close_F"].fillna(0.0).to_numpy()
            d = np.diff(s)
            return d[d != 0]  # nur Sprünge (nicht-null)
        return np.array([])
    elif mode == "dot":
        if "dot" in df.columns:
            return df["dot"].fillna(0.0).to_numpy()
        return np.array([])
    elif mode == "cum_height":
        if "cum_height" in df.columns:
            s = df["cum_height"].fillna(0.0).to_numpy()
            return np.diff(s)
        return np.array([])
    return np.array([])


def _rs_single(series: np.ndarray, window: int) -> float:
    """
    Berechnet R/S fuer ein einzelnes Fenster.
    Returns: mittleres R/S ueber alle nicht-ueberlappenden Fenster
    """
    n = len(series)
    n_windows = n // window
    if n_windows == 0:
        return np.nan

    rs_vals = []
    for i in range(n_windows):
        seg = series[i * window:(i + 1) * window].astype(float)
        mean = seg.mean()
        devs = np.cumsum(seg - mean)
        R = devs.max() - devs.min()
        S = seg.std(ddof=1)
        if S > 0:
            rs_vals.append(R / S)

    return np.mean(rs_vals) if rs_vals else np.nan


def compute_hurst_rs(series: np.ndarray,
                     min_window: int = RS_MIN_WINDOW,
                     max_window: int = None,
                     n_windows: int = RS_N_WINDOWS) -> dict:
    """
    Berechnet den Hurst-Exponenten via R/S Analyse.

    Returns dict mit:
        H      : Hurst-Exponent
        windows: verwendete Fenstergroessen
        rs_vals: R/S Werte je Fenster
        r2     : Bestimmtheitsmass der Regression
    """
    n = len(series)
    if n < RS_MIN_LENGTH:
        return {"H": np.nan, "windows": [], "rs_vals": [], "r2": np.nan}

    if max_window is None:
        max_window = n // 4

    max_window = max(max_window, min_window * 2)

    windows = np.unique(
        np.logspace(
            np.log10(min_window),
            np.log10(max_window),
            n_windows
        ).astype(int)
    )
    windows = windows[windows >= min_window]
    windows = windows[windows <= n]

    rs_vals = []
    valid_windows = []
    for w in windows:
        rs = _rs_single(series, int(w))
        if not np.isnan(rs) and rs > 0:
            rs_vals.append(rs)
            valid_windows.append(w)

    if len(valid_windows) < 3:
        return {"H": np.nan, "windows": [], "rs_vals": [], "r2": np.nan}

    log_w  = np.log10(valid_windows)
    log_rs = np.log10(rs_vals)

    # Lineare Regression: log(R/S) = H * log(n) + const
    coeffs = np.polyfit(log_w, log_rs, 1)
    H = coeffs[0]

    # R² berechnen
    y_pred = np.polyval(coeffs, log_w)
    ss_res = np.sum((log_rs - y_pred) ** 2)
    ss_tot = np.sum((log_rs - log_rs.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "H":       float(np.clip(H, 0.0, 1.0)),
        "H_raw":   float(H),
        "windows": valid_windows,
        "rs_vals": rs_vals,
        "r2":      float(r2),
        "n":       n
    }


def run_rs(name: str, ur, fractals: dict, out_dir: Path) -> dict:
    """
    Hauptfunktion — berechnet R/S Hurst je aktivem Fraktal.

    Returns: dict {F: {"H": ..., "r2": ..., ...}}
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    print(f"[rs] R/S Hurst-Analyse fuer {name}")
    print(f"     Input: {RS_INPUT} | Fenster: {RS_MIN_WINDOW}-auto | N={RS_N_WINDOWS}")

    for F in sorted(fractals.keys()):
        df = fractals[F].reset_index(drop=True)

        # Fraktal filtern (gleiche Logik wie viz_panels)
        if "close_F" in df.columns and df["close_F"].nunique() <= 1:
            continue
        if "c_macd" in df.columns:
            if df["c_macd"].abs().max() < F * 0.001:
                continue

        series = _get_series(df, F, RS_INPUT)
        if len(series) < RS_MIN_LENGTH:
            print(f"  F={int(F):6d}: zu wenig Daten ({len(series)} Punkte)")
            continue

        result = compute_hurst_rs(series,
                                  min_window=RS_MIN_WINDOW,
                                  max_window=RS_MAX_WINDOW,
                                  n_windows=RS_N_WINDOWS)
        result["F"] = F
        results[F] = result

        h_str = f"{result['H']:.3f}" if not np.isnan(result['H']) else "n/a"
        r2_str = f"{result['r2']:.3f}" if not np.isnan(result['r2']) else "n/a"
        char = "persistiv" if result['H'] > 0.55 else "Zufall" if result['H'] > 0.45 else "antipersistiv"
        print(f"  F={int(F):6d}: H={h_str}  R²={r2_str}  [{char}]  n={result['n']:,}")

    # Als CSV speichern fuer spaetere Verwendung
    if results:
        rows = []
        for F, r in results.items():
            rows.append({
                "F":     int(F),
                "H":     r["H"],
                "H_raw": r.get("H_raw", r["H"]),
                "r2":    r["r2"],
                "n":     r["n"]
            })
        pd.DataFrame(rows).to_csv(
            out_dir / f"{name}_rs_hurst.csv", index=False)
        print(f"[rs] Gespeichert: {name}_rs_hurst.csv")

    return results