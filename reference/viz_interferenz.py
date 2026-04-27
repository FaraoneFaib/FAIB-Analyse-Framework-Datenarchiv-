"""
FAIB NEXUS - viz_interferenz.py
================================
Fraktal-Interferenz: UP/DOWN binaer

cum_height > close_F + F/2 -> UP   (1)
cum_height < close_F + F/2 -> DOWN (0)

Jeder Wechsel = Superpositions-Durchlauf
(Preis hat die Fraktal-Mitte gekreuzt)

Aktivierung in main.py:
    ENABLE_VIZ_INTERFERENZ = True
    import viz_interferenz
    viz_interferenz.plot_interferenz(name, ur, fractals, out_dir)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


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


def _calc_state(df: pd.DataFrame, F: float) -> np.ndarray:
    """
    Binaerer Zustand basierend auf absolutem Preis vs Fraktal-Mitte:

    Fraktal-Mitte = close_F + F/2
    cum_height = absoluter Preis (normiert)

    cum_height > Mitte -> UP   (1)
    cum_height < Mitte -> DOWN (0)

    Jeder Wechsel = Superpositions-Durchlauf
    (Preis hat die Mitte der aktuellen Fraktal-Stufe gekreuzt)
    """
    if "close_F" not in df.columns or "cum_height" not in df.columns:
        # Fallback: dot-basiert
        if "dot" in df.columns:
            dot = df["dot"].fillna(0.0).to_numpy()
            return (dot >= float(F) / 2.0).astype(int)
        return np.zeros(len(df), dtype=int)

    close_f  = df["close_F"].fillna(0.0).to_numpy()
    cum_h    = df["cum_height"].fillna(0.0).to_numpy()
    mitte    = close_f + float(F) / 2.0

    # Preis oberhalb der Fraktal-Mitte = UP
    return (cum_h > mitte).astype(int)


def plot_interferenz(name, ur, fractals, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    valid_F = _get_valid_fractals(fractals)
    if not valid_F:
        print("[viz_interferenz] Keine aktiven Fraktale!")
        return

    valid_F_rev = list(reversed(valid_F))
    n_F = len(valid_F)
    max_pts = 2000

    print(f"[viz_interferenz] Fraktale: {valid_F}")

    # Referenzlaenge
    n_ref = max_pts

    # Zustaende berechnen
    states = {}
    for F in valid_F:
        df_f = fractals[F].reset_index(drop=True)
        if len(df_f) > max_pts:
            step = max(1, len(df_f) // max_pts)
            df_s = df_f.iloc[::step].iloc[:max_pts].reset_index(drop=True)
        else:
            df_s = df_f.iloc[:max_pts].reset_index(drop=True)
        n_ref = min(n_ref, len(df_s))
        states[F] = _calc_state(df_s, F)

    for F in valid_F:
        states[F] = states[F][:n_ref]

    x = np.arange(n_ref)

    # Wechsel zaehlen (Superpositions-Durchlaeufe)
    wechsel = {}
    for F in valid_F:
        wechsel[F] = int(np.sum(np.abs(np.diff(states[F]))))

    # close_F Wechsel zaehlen — Gegencheck
    close_F_wechsel = {}
    for F in valid_F:
        df_f = fractals[F].reset_index(drop=True)
        if "close_F" in df_f.columns:
            cf = df_f["close_F"].fillna(0.0).to_numpy()
            close_F_wechsel[F] = int(np.sum(np.abs(np.diff(cf)) > 0))
        else:
            close_F_wechsel[F] = 0

    hdr = "  {:>6} | {:>5} | {:>6} | {:>12} | {:>15} | {:>12}".format(
        "F", "UP%", "DOWN%", "Mitte-Kreuz", "close_F Stufen", "Verhaeltnis")
    sep = "  " + "-"*6 + "-+-" + "-"*5 + "-+-" + "-"*6 + "-+-" + "-"*12 + "-+-" + "-"*15 + "-+-" + "-"*12
    print("[viz_interferenz] Wechsel-Analyse (Superpositions-Durchlaeufe vs close_F Stufen):")
    print(hdr)
    print(sep)
    for F in valid_F:
        s = states[F]
        up_pct   = s.mean() * 100
        down_pct = 100 - up_pct
        cf_w = close_F_wechsel[F]
        mid_w = wechsel[F]
        # Verhaeltnis: Mitte-Kreuzungen pro Stufenwechsel
        ratio = mid_w / cf_w if cf_w > 0 else 0
        print(f"  F={int(F):4d} | {up_pct:5.0f}% | {down_pct:5.0f}% | {mid_w:12d} | {cf_w:15d} | {ratio:12.2f}")

    print("  Interpretation:")
    print("  Verhaeltnis hoch = viele Mitte-Kreuzungen pro Stufenwechsel")
    print("  Verhaeltnis 1.0  = jeder Stufenwechsel = eine Mitte-Kreuzung")
    print("  Verhaeltnis < 1  = Stufen wechseln schneller als Mitte gekreuzt wird")

    # cum_height
    ur_reset = ur.reset_index(drop=True)
    if len(ur_reset) > n_ref:
        step = max(1, len(ur_reset) // n_ref)
        ur_plot = ur_reset.iloc[::step].iloc[:n_ref].reset_index(drop=True)
    else:
        ur_plot = ur_reset.iloc[:n_ref].reset_index(drop=True)

    # close_F
    fraktal_colors = ["#ff4444","#ff8800","#ffcc00","#88ff00",
                      "#00ffcc","#00aaff","#aa44ff","#ff44aa"]
    close_F_data = {}
    for i, F in enumerate(valid_F):
        df_f = fractals[F].reset_index(drop=True)
        if len(df_f) > n_ref:
            step = max(1, len(df_f) // n_ref)
            df_s = df_f.iloc[::step].iloc[:n_ref].reset_index(drop=True)
        else:
            df_s = df_f.iloc[:n_ref].reset_index(drop=True)
        if "close_F" in df_s.columns:
            close_F_data[F] = df_s["close_F"].fillna(0.0).to_numpy()

    # Heatmap Matrix
    heat_matrix = np.zeros((n_F, n_ref))
    for i, F in enumerate(valid_F_rev):
        heat_matrix[i, :] = states[F]

    # Interferenz-Summe: UP=+1, DOWN=-1
    interference = np.zeros(n_ref)
    for F in valid_F:
        interference += states[F].astype(float) * 2 - 1
    interference_norm = interference / n_F

    # Wechsel-Dichte (rollierendes Fenster)
    window = max(20, n_ref // 50)
    wechsel_dichte = np.zeros(n_ref)
    for F in valid_F:
        s = states[F]
        w = np.abs(np.diff(s, prepend=s[0])).astype(float)
        kernel = np.ones(window) / window
        wechsel_dichte += np.convolve(w, kernel, mode="same")
    wechsel_dichte /= n_F

    # ── Layout ──────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 19), facecolor="#0a0a0a")
    gs = gridspec.GridSpec(5, 1,
                           height_ratios=[2.0, 2.5, 1.2, 0.9, 1.2],
                           hspace=0.35, figure=fig)

    ax_price  = fig.add_subplot(gs[0])
    ax_heat   = fig.add_subplot(gs[1])
    ax_ratio  = fig.add_subplot(gs[2])
    ax_wdicht = fig.add_subplot(gs[3])
    ax_interf = fig.add_subplot(gs[4])

    def style(ax, title):
        ax.set_facecolor("#111111")
        ax.set_title(title, color="#cccccc", fontsize=9, pad=4)
        ax.grid(True, alpha=0.12, color="#333333")
        ax.tick_params(colors="#666666", labelsize=7)
        for sp in ax.spines.values():
            sp.set_color("#333333")

    # Panel 1: cum_height + close_F
    style(ax_price, f"{name} - cum_height + Fraktal-Closes")
    if "cum_height" in ur_plot.columns:
        ax_price.plot(x, ur_plot["cum_height"].fillna(0.0).to_numpy(),
                      color="white", lw=1.2, label="cum_height", zorder=5)
    for i, F in enumerate(valid_F):
        if F in close_F_data:
            ax_price.step(x, close_F_data[F], where="post",
                         color=fraktal_colors[i % len(fraktal_colors)],
                         alpha=0.7, lw=0.9, label=f"F={int(F)}")
    ax_price.legend(ncol=5, fontsize=7, facecolor="#1a1a1a",
                    edgecolor="#333333", labelcolor="#aaaaaa", loc="upper right")
    ax_price.set_ylabel("cum_height", color="#888888", fontsize=7)

    # Panel 2: Heatmap
    style(ax_heat, "Fraktal-Interferenz: UP (gruen) / DOWN (rot) | Rechts: Wechsel = Superpositions-Durchlaeufe")
    cmap_heat = mcolors.ListedColormap(["#8b1a1a", "#1a5a1a"])
    ax_heat.imshow(heat_matrix, aspect="auto", interpolation="nearest",
                   cmap=cmap_heat, vmin=0, vmax=1)

    ax_heat.set_yticks(range(n_F))
    ax_heat.set_yticklabels([f"F={int(F)}" for F in valid_F_rev],
                             fontsize=7, color="#aaaaaa")

    # Wechsel rechts
    ax_r = ax_heat.twinx()
    ax_r.set_ylim(ax_heat.get_ylim())
    ax_r.set_yticks(range(n_F))
    ax_r.set_yticklabels([f"{wechsel[F]}x" for F in valid_F_rev],
                          fontsize=7, color="#ffaa44")
    ax_r.tick_params(colors="#555555")

    # Trennlinie gross/klein
    half_idx = n_F // 2
    ax_heat.axhline(half_idx - 0.5, color="#ffff00", lw=1.0, alpha=0.6, ls="--")

    ax_heat.legend(handles=[
        Patch(facecolor="#1a5a1a", label="UP   (cum_height > close_F + F/2)"),
        Patch(facecolor="#8b1a1a", label="DOWN (cum_height < close_F + F/2)"),
    ], loc="upper right", fontsize=7, facecolor="#1a1a1a",
       edgecolor="#333333", labelcolor="#aaaaaa")
    ax_heat.set_ylabel("Fraktal", color="#888888", fontsize=7)
    ax_heat.set_xticks([])

    # Panel 3: Resonanz-Verhaeltnis
    style(ax_ratio, "Vibrations-Intensitaet (Wechsel / theoretisches Maximum) — je hoeher desto resonanter")

    ratios    = []
    f_labels  = []
    bar_colors = []

    # Normierung: Wechsel / theoretisches Maximum (= n_ref Ticks)
    # 0.0 = kein Wechsel, 1.0 = jeder Tick ein Wechsel (maximale Vibration)
    for F in valid_F_rev:
        mid_w = wechsel[F]
        ratio = mid_w / n_ref  # relativ zum theoretischen Maximum
        ratios.append(ratio)
        f_labels.append(f"F={int(F)}")
        # Farbe: hoehere Werte = mehr Vibration = mehr blau/gruen
        if ratio >= 0.3:
            bar_colors.append("#00cc00")   # hoch = resonant
        elif ratio >= 0.15:
            bar_colors.append("#1a6faf")   # mittel
        else:
            bar_colors.append("#b22222")   # niedrig = traege

    y_pos = np.arange(len(valid_F_rev))
    bars = ax_ratio.barh(y_pos, ratios, color=bar_colors, alpha=0.8, height=0.6)

    # Referenzlinien
    ax_ratio.axvline(0.5, color="#ffff00", lw=1.5, ls="--", alpha=0.8,
                     label="50% Maximum")
    ax_ratio.axvline(0.25, color="#888888", lw=0.8, ls=":", alpha=0.5,
                     label="25% Maximum")

    ax_ratio.set_yticks(y_pos)
    ax_ratio.set_yticklabels(f_labels, fontsize=7, color="#aaaaaa")
    ax_ratio.set_xlabel("Wechsel / theoretisches Maximum (n_ref Ticks)", color="#888888", fontsize=7)
    ax_ratio.set_xlim(0, 1.0)

    # Werte an den Balken
    for i, (bar, ratio) in enumerate(zip(bars, ratios)):
        col = "white" if ratio > 0.05 else "#888888"
        ax_ratio.text(ratio + 0.01, i, f"{ratio:.3f}  ({int(ratio*n_ref)}x)",
                      va="center", fontsize=7, color=col)

    ax_ratio.legend(fontsize=7, facecolor="#1a1a1a",
                    edgecolor="#333333", labelcolor="#aaaaaa")

    # Trennlinie gross/klein
    ax_ratio.axhline(n_F // 2 - 0.5, color="#ffff00", lw=0.8, alpha=0.4, ls="--")

    # Panel 3: Wechsel-Dichte
    style(ax_wdicht, f"Wechsel-Dichte (Fenster={window}) — hoch = Markt unentschieden, viele Superpositions-Durchlaeufe")
    ax_wdicht.fill_between(x, wechsel_dichte, 0, color="#7f77dd", alpha=0.5)
    ax_wdicht.plot(x, wechsel_dichte, color="#afa9ec", lw=0.7)
    ax_wdicht.set_ylabel("Wechsel/Tick", color="#888888", fontsize=7)
    ax_wdicht.set_xticks([])

    # Panel 4: Interferenz-Summe
    style(ax_interf, "Interferenz-Summe: +1=alle UP (konstruktiv Long), -1=alle DOWN (konstruktiv Short)")
    pos_m = interference_norm >= 0
    neg_m = interference_norm < 0
    ax_interf.fill_between(x, interference_norm, 0, where=pos_m,
                            color="#1d9e75", alpha=0.7, label="Konstruktiv UP")
    ax_interf.fill_between(x, interference_norm, 0, where=neg_m,
                            color="#d85a30", alpha=0.7, label="Konstruktiv DOWN")
    ax_interf.plot(x, interference_norm, color="white", lw=0.5, alpha=0.4)
    ax_interf.axhline(0,    color="#555555", lw=1.0)
    ax_interf.axhline( 0.5, color="#1d9e75", lw=0.5, ls="--", alpha=0.4)
    ax_interf.axhline(-0.5, color="#d85a30", lw=0.5, ls="--", alpha=0.4)
    ax_interf.set_ylim(-1.1, 1.1)
    ax_interf.set_ylabel("Interferenz", color="#888888", fontsize=7)
    ax_interf.set_xlabel("Tick-Index (gesampelt)", color="#888888", fontsize=7)
    ax_interf.legend(fontsize=7, facecolor="#1a1a1a",
                     edgecolor="#333333", labelcolor="#aaaaaa", loc="upper right")

    for ax in [ax_price, ax_heat, ax_wdicht, ax_interf]:
        ax.set_xlim(0, n_ref)

    fig.suptitle(
        f"FAIB Fraktal-Interferenz - {name}\n"
        f"Aktive Fraktale: {[int(f) for f in valid_F]} | "
        f"F={int(valid_F[-1])} vibriert ({wechsel[valid_F[-1]]}x) | "
        f"F={int(valid_F[0])} traege ({wechsel[valid_F[0]]}x)",
        color="#cccccc", fontsize=10, y=0.998
    )

    plt.tight_layout()
    out_path = out_dir / f"{name}_interferenz.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor="#0a0a0a")
    plt.close(fig)
    print(f"[viz_interferenz] Gespeichert: {out_path}")