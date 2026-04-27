"""
FAIB NEXUS — viz_hurst.py (v2)
================================
Visualisierung: Hurst / DFA / MFDFA

Layout:
- Panel 1 (oben):  cum_height + close_F Treppen
- Panel 2 (unten links):  Rollierender R/S je Fraktal
- Panel 3 (unten mitte):  Rollierender DFA je Fraktal
- Panel 4 (unten rechts): MFDFA h(q) Heatmap (F x q)

Aktivierung in main.py:
    ENABLE_RS = True
    ENABLE_DFA = True
    ENABLE_MFDFA = True
    ENABLE_VIZ_HURST = True
"""

# ============================================================
# PARAMETER
# ============================================================

# Rollierende Fenstergroesse fuer R/S und DFA
VH_ROLLING_WINDOW = 200    # Ticks pro Fenster
VH_ROLLING_STEP   = 20     # Schrittweite (Performance)

# Sampling fuer Plot (max Punkte)
VH_MAX_PLOT_PTS   = 2000

# Farben je Fraktal
VH_COLORS = [
    "#ff4444", "#ff8800", "#ffcc00", "#88ff00",
    "#00ffcc", "#00aaff", "#aa44ff", "#ff44aa",
    "#ffffff", "#aaaaaa", "#555555"
]

# Referenzlinien
VH_H_REF    = 0.5   # Zufalls-Grenze
VH_ALPHA_REF = [0.5, 1.0]

# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


def _get_valid_fractals(fractals: dict) -> list:
    valid = []
    for F in sorted(fractals.keys()):
        df = fractals[F]
        if "close_F" in df.columns and df["close_F"].nunique() <= 1:
            continue
        if "c_macd" in df.columns:
            if df["c_macd"].abs().max() < F * 0.001:
                continue
        valid.append(F)
    return valid


def _sample(arr, max_pts=VH_MAX_PLOT_PTS):
    n = len(arr)
    if n <= max_pts:
        return arr
    step = max(1, n // max_pts)
    return arr[::step][:max_pts]


def _rolling_rs(series: np.ndarray, window: int, step: int) -> np.ndarray:
    """Rollierender R/S Hurst."""
    n = len(series)
    result = np.full(n, np.nan)
    for i in range(window, n, step):
        seg = series[i-window:i]
        seg = seg[~np.isnan(seg)]
        if len(seg) < 20:
            continue
        # Einfacher R/S
        mean = seg.mean()
        devs = np.cumsum(seg - mean)
        R = devs.max() - devs.min()
        S = seg.std(ddof=1)
        if S > 0:
            # Heuristik: H aus einzelnem R/S Wert
            H = np.log(R/S) / np.log(window)
            result[i] = float(np.clip(H, 0.0, 1.0))
    return result


def _rolling_dfa(series: np.ndarray, window: int, step: int,
                  order: int = 1) -> np.ndarray:
    """Rollierender DFA alpha."""
    n = len(series)
    result = np.full(n, np.nan)

    # Fenstergroessen fuer lokale DFA
    min_w = max(8, window // 20)
    max_w = window // 4
    if max_w <= min_w:
        return result

    n_wins = 8
    sub_windows = np.unique(
        np.logspace(np.log10(min_w), np.log10(max_w), n_wins).astype(int)
    )
    sub_windows = sub_windows[sub_windows >= min_w]

    for i in range(window, n, step):
        seg = series[i-window:i]
        seg = seg[~np.isnan(seg)]
        if len(seg) < window // 2:
            continue

        profile = np.cumsum(seg - seg.mean())
        F_vals = []
        valid_w = []

        for w in sub_windows:
            if w >= len(seg):
                continue
            n_seg = len(seg) // w
            if n_seg == 0:
                continue
            flucts = []
            for j in range(n_seg):
                chunk = profile[j*w:(j+1)*w]
                x = np.arange(len(chunk))
                try:
                    c = np.polyfit(x, chunk, order)
                    res = chunk - np.polyval(c, x)
                    flucts.append(np.sqrt(np.mean(res**2)))
                except Exception:
                    pass
            if flucts:
                F_vals.append(np.mean(flucts))
                valid_w.append(w)

        if len(valid_w) >= 3:
            log_w = np.log10(valid_w)
            log_f = np.log10(F_vals)
            try:
                alpha = np.polyfit(log_w, log_f, 1)[0]
                # Ausreisser entfernen: nur plausible Werte
                if 0.1 <= alpha <= 1.5:
                    result[i] = float(alpha)
            except Exception:
                pass

    return result


def _style(ax, title, ylabel=""):
    ax.set_facecolor("#111111")
    ax.set_title(title, color="#cccccc", fontsize=8, pad=4)
    ax.tick_params(colors="#666666", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#333333")
    ax.grid(True, alpha=0.12, color="#333333")
    if ylabel:
        ax.set_ylabel(ylabel, color="#888888", fontsize=7)


def plot_hurst(name, ur, fractals,
               rs_results=None, dfa_results=None, mfdfa_results=None,
               out_dir=None):

    if out_dir is None:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    rs_results    = rs_results    or {}
    dfa_results   = dfa_results   or {}
    mfdfa_results = mfdfa_results or {}

    valid_F = _get_valid_fractals(fractals)
    if not valid_F:
        print("[viz_hurst] Keine aktiven Fraktale!")
        return

    n_F = len(valid_F)
    print(f"[viz_hurst] Berechne rollierenden R/S + DFA fuer {n_F} Fraktale ...")
    print(f"            Fenster={VH_ROLLING_WINDOW} | Schritt={VH_ROLLING_STEP}")

    # ── Sampling Referenz ────────────────────────────────────
    max_pts = VH_MAX_PLOT_PTS
    ref_F   = valid_F[0]
    df_ref  = fractals[ref_F].reset_index(drop=True)
    n_ref   = min(max_pts, len(df_ref))

    # ── cum_height samplen ───────────────────────────────────
    ur_reset = ur.reset_index(drop=True)
    if len(ur_reset) > n_ref:
        step_ur = max(1, len(ur_reset) // n_ref)
        ur_plot = ur_reset.iloc[::step_ur].iloc[:n_ref].reset_index(drop=True)
    else:
        ur_plot = ur_reset.iloc[:n_ref].reset_index(drop=True)

    x = np.arange(n_ref)

    # ── close_F samplen ──────────────────────────────────────
    close_F_data = {}
    for F in valid_F:
        df_f = fractals[F].reset_index(drop=True)
        if len(df_f) > n_ref:
            step_f = max(1, len(df_f) // n_ref)
            df_s = df_f.iloc[::step_f].iloc[:n_ref].reset_index(drop=True)
        else:
            df_s = df_f.iloc[:n_ref].reset_index(drop=True)
        if "close_F" in df_s.columns:
            close_F_data[F] = df_s["close_F"].fillna(0.0).to_numpy()

    # ── Rollierenden R/S + DFA berechnen ────────────────────
    rolling_rs  = {}
    rolling_dfa = {}

    for F in valid_F:
        df_f = fractals[F].reset_index(drop=True)

        # Zeitreihe: diff(close_F) Sprünge
        if "close_F" in df_f.columns:
            s = df_f["close_F"].fillna(0.0).to_numpy()
            series_full = np.diff(s)
        elif "dot" in df_f.columns:
            series_full = df_f["dot"].fillna(0.0).to_numpy()
        else:
            series_full = np.zeros(len(df_f))

        # Auf n_ref samplen
        if len(series_full) > n_ref:
            step_s = max(1, len(series_full) // n_ref)
            series = series_full[::step_s][:n_ref]
        else:
            series = series_full[:n_ref]

        print(f"  F={int(F):6d}: R/S...", end=" ")
        rolling_rs[F]  = _rolling_rs(series, VH_ROLLING_WINDOW, VH_ROLLING_STEP)
        print(f"DFA...", end=" ")
        rolling_dfa[F] = _rolling_dfa(series, VH_ROLLING_WINDOW, VH_ROLLING_STEP)
        print("OK")

    # ── MFDFA Heatmap Matrix ─────────────────────────────────
    # F x q Matrix mit h(q) Werten
    has_mfdfa = len(mfdfa_results) > 0
    if has_mfdfa:
        mfdfa_F = [F for F in valid_F if F in mfdfa_results]
        if mfdfa_F:
            q_vals = mfdfa_results[mfdfa_F[0]]["q"]
            heatmap = np.full((len(mfdfa_F), len(q_vals)), np.nan)
            for i, F in enumerate(mfdfa_F):
                r = mfdfa_results[F]
                hq = r.get("h_q", np.array([]))
                if len(hq) == len(q_vals):
                    heatmap[i, :] = hq

    # ── Layout ──────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 24), facecolor="#0a0a0a")
    gs  = gridspec.GridSpec(
        3, 1,
        height_ratios=[1.0, 1.0, 1.0],
        hspace=0.35,
        figure=fig
    )

    ax_price = fig.add_subplot(gs[0])   # oben: cum_height
    ax_rs    = fig.add_subplot(gs[1])   # mitte: rollierender R/S
    ax_dfa   = fig.add_subplot(gs[2])   # unten: rollierender DFA
    ax_mfdfa = None                     # deaktiviert

    fraktal_colors = VH_COLORS

    # ── Panel 1: cum_height + close_F ───────────────────────
    _style(ax_price,
           f"{name} — cum_height + Fraktal-Closes",
           "cum_height")

    if "cum_height" in ur_plot.columns:
        ax_price.plot(x, ur_plot["cum_height"].fillna(0.0).to_numpy(),
                      color="white", lw=1.2, label="cum_height", zorder=5)

    for i, F in enumerate(valid_F):
        if F in close_F_data:
            ax_price.step(x, close_F_data[F], where="post",
                          color=fraktal_colors[i % len(fraktal_colors)],
                          alpha=0.7, lw=0.8, label=f"F={int(F)}")

    ax_price.legend(ncol=6, fontsize=6, facecolor="#1a1a1a",
                    edgecolor="#333333", labelcolor="#aaaaaa",
                    loc="upper right")
    ax_price.set_xlim(0, n_ref)

    # ── Panel 2: Rollierender R/S ────────────────────────────
    _style(ax_rs,
           f"Rollierender R/S Hurst  (Fenster={VH_ROLLING_WINDOW})",
           "H")

    for i, F in enumerate(valid_F):
        rs_arr = rolling_rs.get(F, np.full(n_ref, np.nan))
        valid  = ~np.isnan(rs_arr)
        if valid.any():
            ax_rs.plot(x[valid], rs_arr[valid],
                       color=fraktal_colors[i % len(fraktal_colors)],
                       lw=0.9, alpha=0.8, label=f"F={int(F)}")

    ax_rs.axhline(VH_H_REF, color="#ffff00", lw=1.0,
                  ls="--", alpha=0.7, label="H=0.5")
    ax_rs.axhline(0.7, color="#1d9e75", lw=0.5,
                  ls=":", alpha=0.4)
    ax_rs.axhline(0.3, color="#d85a30", lw=0.5,
                  ls=":", alpha=0.4)
    ax_rs.set_ylim(0.3, 0.8)
    ax_rs.set_xlim(0, n_ref)
    ax_rs.legend(fontsize=6, facecolor="#1a1a1a",
                 edgecolor="#333333", labelcolor="#aaaaaa",
                 loc="upper right", ncol=2)
    ax_rs.set_xlabel("Tick-Index", color="#888888", fontsize=7)

    # ── Panel 3: Rollierender DFA ────────────────────────────
    _style(ax_dfa,
           f"Rollierender DFA alpha  (Fenster={VH_ROLLING_WINDOW})",
           "alpha")

    for i, F in enumerate(valid_F):
        dfa_arr = rolling_dfa.get(F, np.full(n_ref, np.nan))
        valid   = ~np.isnan(dfa_arr)
        if valid.any():
            ax_dfa.plot(x[valid], dfa_arr[valid],
                        color=fraktal_colors[i % len(fraktal_colors)],
                        lw=0.9, alpha=0.8, label=f"F={int(F)}")

    for ref in VH_ALPHA_REF:
        ls = "--" if ref == 0.5 else "-."
        ax_dfa.axhline(ref, color="#ffff00", lw=1.0,
                       ls=ls, alpha=0.7, label=f"α={ref}")
    ax_dfa.set_ylim(0.2, 1.5)
    ax_dfa.set_xlim(0, n_ref)
    ax_dfa.legend(fontsize=6, facecolor="#1a1a1a",
                  edgecolor="#333333", labelcolor="#aaaaaa",
                  loc="upper right", ncol=2)
    ax_dfa.set_xlabel("Tick-Index (gesampelt)", color="#888888", fontsize=7)
    ax_dfa.set_xlim(0, n_ref)

    # MFDFA Panel deaktiviert — zu instabil fuer Jahresdaten

    fig.suptitle(
        f"FAIB — Hurst / DFA / MFDFA Analyse — {name}\n"
        f"Aktive Fraktale: {[int(f) for f in valid_F]}  |  "
        f"Rollierende Fenster: {VH_ROLLING_WINDOW} Ticks",
        color="#cccccc", fontsize=10, y=0.998
    )

    out_path = out_dir / f"{name}_hurst_dfa_mfdfa.png"
    plt.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor="#0a0a0a")
    plt.close(fig)
    print(f"[viz_hurst] Gespeichert: {out_path}")