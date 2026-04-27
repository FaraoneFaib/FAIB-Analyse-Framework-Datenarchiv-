"""
FAIB NEXUS — dfa.py
====================
Detrended Fluctuation Analysis (DFA)

Robusterer Hurst-Exponent — funktioniert auch bei
nicht-stationaeren Zeitreihen mit Trends.

alpha > 0.5 = persistentes Verhalten
alpha = 0.5 = Zufallsprozess
alpha < 0.5 = antipersistentes Verhalten
alpha = 1.0 = 1/f Rauschen (rosa Rauschen, "lebendig")
alpha = 1.5 = Brown'sches Rauschen (zu viel Struktur)

Aktivierung in main.py:
    ENABLE_DFA = True
    import dfa
    dfa.run_dfa(name, ur, fractals, out_dir)
"""

# ============================================================
# PARAMETER — hier anpassen
# ============================================================

# Minimale Fenstergroesse
DFA_MIN_WINDOW = 8

# Maximale Fenstergroesse (None = automatisch = len/4)
DFA_MAX_WINDOW = None

# Anzahl der logarithmisch verteilten Fenstergroessen
DFA_N_WINDOWS = 20

# DFA Ordnung (Grad des Trendpolynoms):
# 1 = linearer Trend entfernen (DFA-1, Standard)
# 2 = quadratischer Trend (DFA-2, robuster)
# 3 = kubischer Trend (DFA-3, sehr robust)
DFA_ORDER = 1

# Eingangszeitreihe (gleiche Optionen wie rs.py)
# "diff_close" = Differenzen von close_F (empfohlen)
# "dot"        = dot-Werte direkt
# "cum_height" = globale Basisbahn
DFA_INPUT = "diff_close"

# Mindestlaenge
DFA_MIN_LENGTH = 100

# Rollierendes Fenster fuer viz_plotly Panel
# (Anzahl Ticks fuer rollierenden DFA-alpha)
DFA_ROLLING_WINDOW = 500

# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path


def _get_series(df: pd.DataFrame, F: float, mode: str) -> np.ndarray:
    """Extrahiert Eingangszeitreihe."""
    if mode == "diff_close":
        if "close_F" in df.columns:
            s = df["close_F"].fillna(0.0).to_numpy()
            d = np.diff(s)
            return d[d != 0]
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


def _dfa_single(series: np.ndarray, window: int, order: int = 1) -> float:
    """
    Berechnet DFA-Fluktuation F(n) fuer ein einzelnes Fenster.
    Entfernt Polynom-Trend der gegebenen Ordnung.
    """
    n = len(series)
    n_segments = n // window
    if n_segments == 0:
        return np.nan

    # Kumulierte Summe (Profile)
    profile = np.cumsum(series - series.mean())

    fluctuations = []
    for i in range(n_segments):
        seg = profile[i * window:(i + 1) * window]
        x = np.arange(len(seg))
        # Polynom-Fit und Residuen
        try:
            coeffs = np.polyfit(x, seg, order)
            trend  = np.polyval(coeffs, x)
            resid  = seg - trend
            fluctuations.append(np.sqrt(np.mean(resid ** 2)))
        except Exception:
            pass

    return np.mean(fluctuations) if fluctuations else np.nan


def compute_dfa(series: np.ndarray,
                min_window: int = DFA_MIN_WINDOW,
                max_window: int = None,
                n_windows:  int = DFA_N_WINDOWS,
                order:      int = DFA_ORDER) -> dict:
    """
    Berechnet DFA alpha-Exponent.

    Returns dict mit:
        alpha  : DFA Exponent (≈ Hurst)
        windows: verwendete Fenstergroessen
        F_vals : Fluktuationswerte F(n)
        r2     : Bestimmtheitsmass
    """
    n = len(series)
    if n < DFA_MIN_LENGTH:
        return {"alpha": np.nan, "windows": [], "F_vals": [], "r2": np.nan}

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

    F_vals = []
    valid_windows = []
    for w in windows:
        f = _dfa_single(series, int(w), order)
        if not np.isnan(f) and f > 0:
            F_vals.append(f)
            valid_windows.append(w)

    if len(valid_windows) < 3:
        return {"alpha": np.nan, "windows": [], "F_vals": [], "r2": np.nan}

    log_w = np.log10(valid_windows)
    log_f = np.log10(F_vals)

    coeffs = np.polyfit(log_w, log_f, 1)
    alpha  = coeffs[0]

    y_pred = np.polyval(coeffs, log_w)
    ss_res = np.sum((log_f - y_pred) ** 2)
    ss_tot = np.sum((log_f - log_f.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "alpha":   float(np.clip(alpha, 0.0, 2.0)),
        "alpha_raw": float(alpha),
        "windows": valid_windows,
        "F_vals":  F_vals,
        "r2":      float(r2),
        "n":       n
    }


def compute_dfa_rolling(series: np.ndarray,
                        window: int = DFA_ROLLING_WINDOW,
                        order:  int = DFA_ORDER) -> np.ndarray:
    """
    Rollierender DFA alpha — fuer viz_plotly Panel.
    Berechnet alpha in rollierendem Fenster.
    Returns: alpha-Array gleicher Laenge wie series.
    """
    n = len(series)
    alphas = np.full(n, np.nan)

    for i in range(window, n):
        seg = series[i - window:i]
        result = compute_dfa(seg,
                             min_window=max(8, window // 20),
                             max_window=window // 4,
                             n_windows=10,
                             order=order)
        alphas[i] = result["alpha"]

    return alphas


def run_dfa(name: str, ur, fractals: dict, out_dir: Path) -> dict:
    """
    Hauptfunktion — berechnet DFA alpha je aktivem Fraktal.
    Returns: dict {F: {"alpha": ..., "r2": ..., ...}}
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    print(f"[dfa] DFA Analyse fuer {name}")
    print(f"      Input: {DFA_INPUT} | Order: {DFA_ORDER} | Fenster: {DFA_MIN_WINDOW}-auto")

    for F in sorted(fractals.keys()):
        df = fractals[F].reset_index(drop=True)

        if "close_F" in df.columns and df["close_F"].nunique() <= 1:
            continue
        if "c_macd" in df.columns:
            if df["c_macd"].abs().max() < F * 0.001:
                continue

        series = _get_series(df, F, DFA_INPUT)
        if len(series) < DFA_MIN_LENGTH:
            print(f"  F={int(F):6d}: zu wenig Daten ({len(series)} Punkte)")
            continue

        result = compute_dfa(series,
                             min_window=DFA_MIN_WINDOW,
                             max_window=DFA_MAX_WINDOW,
                             n_windows=DFA_N_WINDOWS,
                             order=DFA_ORDER)
        result["F"] = F
        results[F] = result

        a_str  = f"{result['alpha']:.3f}" if not np.isnan(result['alpha']) else "n/a"
        r2_str = f"{result['r2']:.3f}"   if not np.isnan(result['r2'])    else "n/a"

        # Charakter bestimmen
        a = result['alpha']
        if np.isnan(a):
            char = "n/a"
        elif a > 1.3:
            char = "Brown (zu viel Struktur)"
        elif a > 0.9:
            char = "1/f Rosa (lebendig!)"
        elif a > 0.55:
            char = "persistiv"
        elif a > 0.45:
            char = "Zufall"
        else:
            char = "antipersistiv"

        print(f"  F={int(F):6d}: alpha={a_str}  R²={r2_str}  [{char}]  n={result['n']:,}")

    # Speichern
    if results:
        rows = []
        for F, r in results.items():
            rows.append({
                "F":         int(F),
                "alpha":     r["alpha"],
                "alpha_raw": r.get("alpha_raw", r["alpha"]),
                "r2":        r["r2"],
                "n":         r["n"]
            })
        pd.DataFrame(rows).to_csv(
            out_dir / f"{name}_dfa.csv", index=False)
        print(f"[dfa] Gespeichert: {name}_dfa.csv")

    return results