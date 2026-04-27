"""
FAIB NEXUS — viz_verlagerung.py
================================
Test-Modul: Verlagerungs-Varianten vergleichen

Zeigt verschiedene Arten die Verlagerung zu berechnen:
- Alle aktiven Fraktale
- Untere Hälfte (kleine Fraktale)
- Mitte (entscheidender Punkt)
- Obere Hälfte (große Fraktale)
- Nur größter aktiver Fraktal

Aktivierung in main.py:
    ENABLE_VIZ_VERLAGERUNG = True

    try:
        import viz_verlagerung
    except ImportError:
        viz_verlagerung = None

    if ENABLE_VIZ_VERLAGERUNG and viz_verlagerung is not None:
        viz_verlagerung.plot_verlagerung(name, ur, fractals, nexus_df,
                                         fusion_series, verlag_series, out_dir)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def _safe_sample(df, max_points=2500):
    n = len(df)
    if n <= max_points:
        return df.copy()
    step = max(1, n // max_points)
    idx = np.arange(0, n, step)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return df.iloc[idx].copy()


def _get_valid_fractals(fractals: dict) -> list:
    """Gibt gefilterte Fraktale zurück — gleiche Logik wie viz_panels."""
    valid = []
    for F in sorted(fractals.keys()):
        df_check = fractals[F]
        if "close_F" in df_check.columns:
            if df_check["close_F"].nunique() <= 1:
                continue
        if "c_macd" in df_check.columns:
            macd_range = df_check["c_macd"].abs().max()
            if macd_range < F * 0.001:
                continue
        valid.append(F)
    return valid


def _build_fusion_for_subset(fractals: dict, subset: list, n: int) -> pd.Series:
    """
    Berechnet Fusion + Verlagerung nur für eine Teilmenge von Fraktalen.
    """
    if not subset:
        return pd.Series(np.zeros(n))

    # Dominanz je Fraktal berechnen
    dom_list = []
    for F in subset:
        df = fractals[F]
        if "c_macd" in df.columns and "c_signal" in df.columns:
            dom = (df["c_macd"].fillna(0.0) >= df["c_signal"].fillna(0.0)).astype(int)
        else:
            dom = pd.Series(np.zeros(len(df)))
        dom_list.append(dom.reset_index(drop=True))

    # Fusion = Summe (+1 für Long, -1 für Short)
    combined = pd.concat(dom_list, axis=1)
    combined.columns = [f"F{int(F)}" for F in subset]
    signed = combined.replace({1: 1, 0: -1})
    fusion = signed.sum(axis=1)

    # Verlagerung = Ableitung
    verlagerung = fusion.diff().fillna(0.0)
    return verlagerung


def plot_verlagerung(name, ur, fractals, nexus_df, fusion_series, verlag_series, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Aktive Fraktale bestimmen
    valid_F = _get_valid_fractals(fractals)
    n_valid = len(valid_F)

    if n_valid == 0:
        print("[viz_verlagerung] Keine aktiven Fraktale!")
        return

    print(f"[viz_verlagerung] Aktive Fraktale: {valid_F}")

    # Teilmengen definieren
    # Untere Hälfte
    half = n_valid // 2
    subset_klein  = valid_F[:half] if half > 0 else valid_F[:1]

    # Obere Hälfte
    subset_gross  = valid_F[half:] if half > 0 else valid_F[-1:]

    # Mitte — entscheidender Punkt
    mid_idx = n_valid // 2
    if n_valid % 2 == 1:
        # ungerade → genau ein Mittelpunkt
        subset_mitte = [valid_F[mid_idx]]
    else:
        # gerade → zwei mittlere
        subset_mitte = [valid_F[mid_idx - 1], valid_F[mid_idx]]

    # Nur größter
    subset_groesster = [valid_F[-1]]

    print(f"  Klein:     {subset_klein}")
    print(f"  Mitte:     {subset_mitte}")
    print(f"  Groß:      {subset_gross}")
    print(f"  Größster:  {subset_groesster}")

    # Referenz-Länge bestimmen (aus erstem Fraktal)
    n_ref = len(fractals[valid_F[0]])

    # Verlagerungen berechnen
    verl_alle     = _build_fusion_for_subset(fractals, valid_F, n_ref)
    verl_klein    = _build_fusion_for_subset(fractals, subset_klein, n_ref)
    verl_mitte    = _build_fusion_for_subset(fractals, subset_mitte, n_ref)
    verl_gross    = _build_fusion_for_subset(fractals, subset_gross, n_ref)
    verl_groesster = _build_fusion_for_subset(fractals, subset_groesster, n_ref)

    # Sampling — alle auf gleiche Länge
    max_pts = 2500

    def sample_series(s):
        if len(s) <= max_pts:
            return s.reset_index(drop=True)
        step = max(1, len(s) // max_pts)
        return s.iloc[::step].iloc[:max_pts].reset_index(drop=True)

    verl_alle_s      = sample_series(verl_alle)
    verl_klein_s     = sample_series(verl_klein)
    verl_mitte_s     = sample_series(verl_mitte)
    verl_gross_s     = sample_series(verl_gross)
    verl_groesster_s = sample_series(verl_groesster)
    n_plot = len(verl_alle_s)
    x = np.arange(n_plot)

    # cum_height + close_F samplen
    ur_reset = ur.reset_index(drop=True)
    if len(ur_reset) > n_plot:
        step_ur = max(1, len(ur_reset) // n_plot)
        ur_plot = ur_reset.iloc[::step_ur].iloc[:n_plot].reset_index(drop=True)
    else:
        ur_plot = ur_reset.iloc[:n_plot].reset_index(drop=True)

    fraktal_colors = ["tab:red", "tab:blue", "tab:green", "tab:purple",
                      "tab:orange", "tab:brown", "tab:cyan", "tab:olive"]

    # ── Layout ──────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 22), facecolor="#0d0d0d")
    gs = gridspec.GridSpec(6, 1, figure=fig,
                           height_ratios=[2, 1, 1, 1, 1, 1],
                           hspace=0.35)

    axes = [fig.add_subplot(gs[i]) for i in range(6)]

    def style(ax, title):
        ax.set_facecolor("#111111")
        ax.set_title(title, color="#cccccc", fontsize=9, pad=4)
        ax.grid(True, alpha=0.15, color="#333333")
        ax.tick_params(colors="#666666", labelsize=7)
        ax.spines[:].set_color("#333333")
        ax.axhline(0, color="#555555", lw=0.8)

    # ── Panel 0: cum_height + close_F ───────────────────────
    style(axes[0], f"{name} – cum_height + Fraktal-Closes (aktive Ebenen: {valid_F})")
    if "cum_height" in ur_plot.columns:
        axes[0].plot(x, ur_plot["cum_height"].fillna(0.0).to_numpy(),
                     color="white", lw=1.2, label="cum_height", zorder=5)

    for fi, F in enumerate(valid_F):
        dfF = fractals[F].reset_index(drop=True)
        if len(dfF) > n_plot:
            step_f = max(1, len(dfF) // n_plot)
            dfF_plot = dfF.iloc[::step_f].iloc[:n_plot].reset_index(drop=True)
        else:
            dfF_plot = dfF.iloc[:n_plot].reset_index(drop=True)
        if "close_F" in dfF_plot.columns:
            axes[0].step(x, dfF_plot["close_F"].fillna(0.0).to_numpy(),
                         where="post",
                         color=fraktal_colors[fi % len(fraktal_colors)],
                         alpha=0.75, lw=0.9, label=f"F={int(F)}")

    axes[0].legend(ncol=5, fontsize=7, facecolor="#1a1a1a",
                   edgecolor="#333333", labelcolor="#aaaaaa")
    axes[0].set_ylabel("cum_height", color="#888888", fontsize=7)

    # ── Hilfsfunktion für Verlagerungs-Panel ────────────────
    def plot_verl(ax, verl, title, subset_info, color_pos="#1f7a1f", color_neg="#b22222"):
        style(ax, title)
        v = verl.to_numpy()
        colors = [color_pos if val >= 0 else color_neg for val in v]
        ax.bar(x, v, color=colors, alpha=0.8, width=1.0)
        ax.set_ylabel("Verlagerung", color="#888888", fontsize=7)
        # Subset Info als Text
        ax.text(0.99, 0.95, f"Fraktale: {[int(f) for f in subset_info]}",
                transform=ax.transAxes, fontsize=6, color="#666666",
                ha="right", va="top")

    # ── Panel 1: Alle aktiven ────────────────────────────────
    plot_verl(axes[1], verl_alle_s,
              f"Verlagerung ALLE aktiven ({n_valid} Fraktale)",
              valid_F, "#1f7a1f", "#b22222")

    # ── Panel 2: Klein (untere Hälfte) ───────────────────────
    plot_verl(axes[2], verl_klein_s,
              f"Verlagerung KLEIN — untere Hälfte ({len(subset_klein)} Fraktale)",
              subset_klein, "#4488ff", "#b22222")

    # ── Panel 3: Mitte ───────────────────────────────────────
    plot_verl(axes[3], verl_mitte_s,
              f"Verlagerung MITTE — entscheidender Punkt ({len(subset_mitte)} Fraktal{'e' if len(subset_mitte)>1 else ''})",
              subset_mitte, "#ffff00", "#ff8800")

    # ── Panel 4: Groß (obere Hälfte) ─────────────────────────
    plot_verl(axes[4], verl_gross_s,
              f"Verlagerung GROSS — obere Hälfte ({len(subset_gross)} Fraktale)",
              subset_gross, "#e8a010", "#b22222")

    # ── Panel 5: Nur größter ─────────────────────────────────
    plot_verl(axes[5], verl_groesster_s,
              f"Verlagerung NUR GRÖSSTER — F={int(subset_groesster[0])} (Regime-Signal)",
              subset_groesster, "#ff4444", "#4488ff")

    axes[5].set_xlabel("Tick-Index (gesampelt)", color="#888888", fontsize=7)

    fig.suptitle(
        f"{name} – Verlagerungs-Varianten Test\n"
        f"Aktive Fraktale: {[int(f) for f in valid_F]}",
        color="#cccccc", fontsize=11, y=0.995
    )

    out_path = out_dir / f"{name}_verlagerung_varianten.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig)
    print(f"[viz_verlagerung] Gespeichert: {out_path}")