"""
FAIB NEXUS — mfdfa.py
======================
Multifraktale Detrended Fluctuation Analysis (MFDFA)

Berechnet das Hurst-Spektrum h(q) fuer verschiedene Momente q.
Zeigt ob der Markt mono- oder multifraktal ist.

Schmales h(q)-Spektrum  = monofraktal (einfache Struktur)
Breites h(q)-Spektrum   = multifraktal (komplexe Struktur)
h(2)                    = klassischer DFA alpha
delta_h = h_min - h_max = Multifraktalitaets-Breite

Aktivierung in main.py:
    ENABLE_MFDFA = True
    import mfdfa
    mfdfa.run_mfdfa(name, ur, fractals, out_dir)
"""

# ============================================================
# PARAMETER — hier anpassen
# ============================================================

# Momente q — Bereich und Anzahl
# q > 0: grosse Fluktuationen (Extremereignisse)
# q < 0: kleine Fluktuationen (ruhige Phasen)
# q = 2: entspricht klassischem DFA
MFDFA_Q_MIN   = -5       # kleinster Moment
MFDFA_Q_MAX   =  5       # groesster Moment
MFDFA_Q_STEPS = 21       # Anzahl q-Werte (inkl. 0)

# Fenstergroessen
MFDFA_MIN_WINDOW = 8
MFDFA_MAX_WINDOW = None   # None = automatisch len/4
MFDFA_N_WINDOWS  = 15

# DFA Ordnung (Trendpolynom-Grad)
MFDFA_ORDER = 1

# Eingangszeitreihe
# "diff_close" = Differenzen von close_F (empfohlen)
# "dot"        = dot-Werte
MFDFA_INPUT = "diff_close"

# Mindestlaenge
MFDFA_MIN_LENGTH = 200

# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path


def _get_series(df: pd.DataFrame, F: float, mode: str) -> np.ndarray:
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
    return np.array([])


def _mfdfa_fluctuations(series: np.ndarray,
                         window: int,
                         order:  int = 1) -> np.ndarray:
    """
    Berechnet Fluktuationen F_s fuer alle Segmente.
    Returns: Array der Fluktuations-Werte je Segment.
    """
    n = len(series)
    profile = np.cumsum(series - series.mean())
    n_seg = n // window

    if n_seg == 0:
        return np.array([])

    F_s = []
    for i in range(n_seg):
        seg = profile[i * window:(i + 1) * window]
        x   = np.arange(len(seg))
        try:
            coeffs = np.polyfit(x, seg, order)
            trend  = np.polyval(coeffs, x)
            resid  = seg - trend
            F_s.append(np.sqrt(np.mean(resid ** 2)))
        except Exception:
            F_s.append(np.nan)

    return np.array([f for f in F_s if not np.isnan(f) and f > 0])


def compute_mfdfa(series:      np.ndarray,
                  q_values:    np.ndarray = None,
                  min_window:  int = MFDFA_MIN_WINDOW,
                  max_window:  int = None,
                  n_windows:   int = MFDFA_N_WINDOWS,
                  order:       int = MFDFA_ORDER) -> dict:
    """
    Berechnet MFDFA h(q) Spektrum.

    Returns dict mit:
        q      : q-Werte
        h_q    : generalisierte Hurst-Exponenten h(q)
        tau_q  : Skalierungsfunktion tau(q)
        alpha_s: singulaeres Spektrum alpha (Hoelder-Exponenten)
        f_alpha: multifraktales Spektrum f(alpha)
        delta_h: Breite des Spektrums (Multifraktalitaet)
        h2     : h(q=2) ≈ klassischer DFA alpha
        r2_vals: Regressions-Guete je q
    """
    n = len(series)
    if n < MFDFA_MIN_LENGTH:
        return {"h_q": np.array([]), "q": np.array([]),
                "delta_h": np.nan, "h2": np.nan}

    if q_values is None:
        q_values = np.linspace(MFDFA_Q_MIN, MFDFA_Q_MAX, MFDFA_Q_STEPS)

    # q=0 gesondert behandeln (logarithmisches Mittel)
    q_values = q_values[q_values != 0]

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
    windows = windows[windows <= n // 4]

    if len(windows) < 3:
        return {"h_q": np.array([]), "q": q_values,
                "delta_h": np.nan, "h2": np.nan}

    # F_q(s) Matrix: [n_q × n_windows]
    F_q = np.full((len(q_values), len(windows)), np.nan)

    for j, w in enumerate(windows):
        F_s = _mfdfa_fluctuations(series, int(w), order)
        if len(F_s) == 0:
            continue

        for i, q in enumerate(q_values):
            if q > 0:
                F_q[i, j] = np.mean(F_s ** q) ** (1.0 / q)
            else:
                F_q[i, j] = np.mean(F_s ** q) ** (1.0 / q)

    # h(q) durch Regression log(F_q) = h(q) * log(s) + const
    log_w = np.log10(windows)
    h_q   = np.full(len(q_values), np.nan)
    r2_vals = np.full(len(q_values), np.nan)

    for i in range(len(q_values)):
        row = F_q[i, :]
        mask = np.isfinite(row) & (row > 0)
        if mask.sum() < 3:
            continue

        lw = log_w[mask]
        lf = np.log10(row[mask])

        try:
            coeffs = np.polyfit(lw, lf, 1)
            h_q[i] = coeffs[0]

            y_pred = np.polyval(coeffs, lw)
            ss_res = np.sum((lf - y_pred) ** 2)
            ss_tot = np.sum((lf - lf.mean()) ** 2)
            r2_vals[i] = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
        except Exception:
            pass

    # Multifraktales Spektrum
    valid = np.isfinite(h_q)
    delta_h = float(h_q[valid].max() - h_q[valid].min()) if valid.sum() > 1 else np.nan

    # h(q=2) ≈ klassischer DFA alpha
    q2_idx = np.argmin(np.abs(q_values - 2.0))
    h2 = float(h_q[q2_idx]) if np.isfinite(h_q[q2_idx]) else np.nan

    # tau(q) = q * h(q) - 1
    tau_q = q_values * h_q - 1.0

    # Singulaeres Spektrum via Legendre-Transformation
    # alpha = d(tau)/dq, f(alpha) = q*alpha - tau(q)
    alpha_s = np.gradient(tau_q, q_values)
    f_alpha = q_values * alpha_s - tau_q

    return {
        "q":       q_values,
        "h_q":     h_q,
        "tau_q":   tau_q,
        "alpha_s": alpha_s,
        "f_alpha": f_alpha,
        "delta_h": delta_h,
        "h2":      h2,
        "r2_vals": r2_vals,
        "n":       n
    }


def run_mfdfa(name: str, ur, fractals: dict, out_dir: Path) -> dict:
    """
    Hauptfunktion — berechnet MFDFA h(q) je aktivem Fraktal.
    Returns: dict {F: {q, h_q, delta_h, h2, ...}}
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    q_values = np.linspace(MFDFA_Q_MIN, MFDFA_Q_MAX, MFDFA_Q_STEPS)
    q_values = q_values[q_values != 0]

    print(f"[mfdfa] MFDFA Analyse fuer {name}")
    print(f"        q=[{MFDFA_Q_MIN}..{MFDFA_Q_MAX}] ({MFDFA_Q_STEPS} Werte) | Order={MFDFA_ORDER}")

    for F in sorted(fractals.keys()):
        df = fractals[F].reset_index(drop=True)

        if "close_F" in df.columns and df["close_F"].nunique() <= 1:
            continue
        if "c_macd" in df.columns:
            if df["c_macd"].abs().max() < F * 0.001:
                continue

        series = _get_series(df, F, MFDFA_INPUT)
        if len(series) < MFDFA_MIN_LENGTH:
            print(f"  F={int(F):6d}: zu wenig Daten ({len(series)} Punkte)")
            continue

        result = compute_mfdfa(series,
                               q_values=q_values,
                               min_window=MFDFA_MIN_WINDOW,
                               max_window=MFDFA_MAX_WINDOW,
                               n_windows=MFDFA_N_WINDOWS,
                               order=MFDFA_ORDER)
        result["F"] = F
        results[F] = result

        h2_str  = f"{result['h2']:.3f}"    if not np.isnan(result['h2'])    else "n/a"
        dh_str  = f"{result['delta_h']:.3f}" if not np.isnan(result['delta_h']) else "n/a"
        multi = "multifraktal" if (not np.isnan(result['delta_h']) and
                                   result['delta_h'] > 0.2) else "monofraktal"
        print(f"  F={int(F):6d}: h(2)={h2_str}  delta_h={dh_str}  [{multi}]  n={result['n']:,}")

    # Speichern
    if results:
        rows = []
        for F, r in results.items():
            if len(r["h_q"]) > 0:
                for qi, q in enumerate(r["q"]):
                    rows.append({
                        "F":       int(F),
                        "q":       float(q),
                        "h_q":     float(r["h_q"][qi]),
                        "delta_h": r["delta_h"],
                        "h2":      r["h2"]
                    })
        if rows:
            pd.DataFrame(rows).to_csv(
                out_dir / f"{name}_mfdfa.csv", index=False)
            print(f"[mfdfa] Gespeichert: {name}_mfdfa.csv")

    return results