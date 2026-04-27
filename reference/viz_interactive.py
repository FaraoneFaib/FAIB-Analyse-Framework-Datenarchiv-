"""
FAIB – Interaktive Analyse mit verschiebbarer vertikaler Linie
==============================================================
Aktivierung in main.py:
    ENABLE_VIZ_INTERACTIVE = True

    if ENABLE_VIZ_INTERACTIVE and viz_interactive is not None:
        viz_interactive.plot_interactive(name, ur, fractals, nexus_df, out_dir)

Bedienung:
    - Maus über Chart bewegen → vertikale Linie folgt
    - Klick → Linie fixieren / lösen
    - Rechts: Live-Status aller Fraktalebenen
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # robustestes Backend für interaktive Fenster
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
from matplotlib.patches import FancyBboxPatch


def _safe_sample(df, max_points=1800):
    n = len(df)
    if n <= max_points:
        return df.copy()
    step = max(1, n // max_points)
    idx = np.arange(0, n, step)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return df.iloc[idx].copy()


def plot_interactive(name, ur, fractals, nexus_df, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_levels = sorted(fractals.keys())
    if not panel_levels:
        return

    # --- Sampling ---
    sampled = {}
    index_maps = {}
    min_len = None

    for F in panel_levels:
        df_full = fractals[F].copy().reset_index(drop=True)
        df_s = _safe_sample(df_full, max_points=1800)
        sampled[F] = df_s
        index_maps[F] = df_s.index.to_numpy()
        min_len = len(df_s) if min_len is None else min(min_len, len(df_s))

    if not min_len:
        return

    for F in panel_levels:
        sampled[F] = sampled[F].iloc[:min_len].copy().reset_index(drop=True)
        index_maps[F] = index_maps[F][:min_len]

    n_levels = len(panel_levels)
    color_list = ["tab:green","tab:orange","tab:purple","tab:red",
                  "tab:brown","tab:pink","tab:cyan","tab:olive"]

    # --- Daten vorbereiten ---
    data = {}
    for F in panel_levels:
        df = sampled[F]
        c_macd   = df["c_macd"].fillna(0.0).to_numpy()   if "c_macd"   in df.columns else np.zeros(min_len)
        c_signal = df["c_signal"].fillna(0.0).to_numpy() if "c_signal" in df.columns else np.zeros(min_len)
        c_hist   = df["c_hist"].fillna(0.0).to_numpy()   if "c_hist"   in df.columns else np.zeros(min_len)
        dom = np.where(c_macd >= c_signal, 1, 0)
        exp = np.where(c_macd >= 0, 1, 0)
        data[F] = dict(macd=c_macd, signal=c_signal, hist=c_hist, dom=dom, exp=exp)

    bin_matrix  = np.array([data[F]["dom"] for F in panel_levels])
    comp_matrix = np.array([data[F]["exp"] for F in panel_levels])

    x = np.arange(min_len)

    # -------------------------------------------------------
    # Layout
    # -------------------------------------------------------
    fig = plt.figure(figsize=(20, 3.0 * n_levels + 6.5), facecolor="#0d0d0d")
    outer = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[5, 1], wspace=0.03)

    left_gs = gridspec.GridSpecFromSubplotSpec(
        n_levels + 3, 1,
        subplot_spec=outer[0],
        height_ratios=[1.2] + [1.0] * n_levels + [0.9, 0.9],
        hspace=0.25,
    )

    top_ax   = fig.add_subplot(left_gs[0, 0])
    macd_axs = [fig.add_subplot(left_gs[i + 1, 0]) for i in range(n_levels)]
    heat_ax  = fig.add_subplot(left_gs[n_levels + 1, 0])
    comp_ax  = fig.add_subplot(left_gs[n_levels + 2, 0])
    status_ax = fig.add_subplot(outer[1])

    def style_ax(ax):
        ax.set_facecolor("#111111")
        ax.tick_params(colors="#888888", labelsize=7)
        ax.spines[:].set_color("#333333")
        ax.grid(True, alpha=0.12, color="#444444")

    # -------------------------------------------------------
    # TOP PANEL
    # -------------------------------------------------------
    style_ax(top_ax)
    top_colors = {F: color_list[i % len(color_list)] for i, F in enumerate(panel_levels)}

    for F in panel_levels:
        df = sampled[F]
        if "cum_height" in df.columns:
            top_ax.plot(x, df["cum_height"].fillna(0).to_numpy(),
                        lw=0.9, alpha=0.8, color=top_colors[F], label=f"F={F}")
        if "close_F" in df.columns:
            top_ax.plot(x, df["close_F"].fillna(0).to_numpy(),
                        lw=0.8, alpha=0.4, color=top_colors[F], linestyle="--")

    level_str = ",".join(str(F) for F in panel_levels)
    top_ax.set_title(f"cum_height + Fraktal-Close (F={level_str})",
                     color="#cccccc", fontsize=8, pad=4)
    top_ax.set_ylabel("Wert", color="#888888", fontsize=7)
    top_ax.set_xlim(0, min_len - 1)

    handles, labels = top_ax.get_legend_handles_labels()
    if handles:
        top_ax.legend(handles, labels, loc="upper right", fontsize=6, ncol=4,
                      facecolor="#1a1a1a", edgecolor="#333333", labelcolor="#aaaaaa")

    # -------------------------------------------------------
    # MACD PANELS
    # -------------------------------------------------------
    for ax, F in zip(macd_axs, panel_levels):
        style_ax(ax)
        d = data[F]
        hist_pos = np.where(d["hist"] >= 0, d["hist"], np.nan)
        hist_neg = np.where(d["hist"] <  0, d["hist"], np.nan)
        ax.bar(x, hist_pos, color="#1f7a1f", alpha=0.4, width=1.0)
        ax.bar(x, hist_neg, color="#b22222", alpha=0.4, width=1.0)
        ax.plot(x, d["macd"],   color="#4488ff", lw=0.9, label="c_macd")
        ax.plot(x, d["signal"], color="#ffffff", lw=0.8, alpha=0.7, label="c_signal")
        ax.axhline(0, color="#555555", lw=0.7)
        ax.set_title(f"MACD F={F}", color="#aaaaaa", fontsize=7, pad=3)
        ax.set_ylabel("MACD", color="#666666", fontsize=6)
        ax.set_xlim(0, min_len - 1)
        ax.legend(loc="upper right", fontsize=5, facecolor="#1a1a1a",
                  edgecolor="#333333", labelcolor="#aaaaaa")

    macd_axs[-1].set_xlabel("Index", color="#888888", fontsize=7)

    # -------------------------------------------------------
    # HEATMAPS — imshow x-Bereich explizit setzen
    # -------------------------------------------------------
    cmap_dom  = ListedColormap(["#b22222", "#1f7a1f"])
    cmap_comp = ListedColormap(["#1a6faf", "#e8a010"])

    heat_ax.imshow(bin_matrix,  aspect="auto", interpolation="nearest",
                   cmap=cmap_dom,  vmin=0, vmax=1,
                   extent=[0, min_len, -0.5, n_levels - 0.5])
    comp_ax.imshow(comp_matrix, aspect="auto", interpolation="nearest",
                   cmap=cmap_comp, vmin=0, vmax=1,
                   extent=[0, min_len, -0.5, n_levels - 0.5])

    for ax, title in [(heat_ax, "Dominanz (grün=Long, rot=Short)"),
                      (comp_ax, "Expansion (orange) / Kompression (blau)")]:
        ax.set_facecolor("#111111")
        ax.set_title(title, color="#cccccc", fontsize=7, pad=3)
        ax.set_yticks(np.arange(n_levels))
        ax.set_yticklabels([f"F={F}" for F in panel_levels], fontsize=6, color="#aaaaaa")
        ax.set_ylabel("Fraktal", color="#888888", fontsize=6)
        ax.tick_params(colors="#666666")
        ax.set_xlim(0, min_len - 1)

    # -------------------------------------------------------
    # STATUS PANEL
    # -------------------------------------------------------
    status_ax.set_facecolor("#0a0a0a")
    status_ax.set_xlim(0, 1)
    status_ax.set_ylim(0, 1)
    status_ax.axis("off")

    fig.suptitle(
        f"{name} – FAIB Interaktive Analyse  |  Maus bewegen → Linie folgt  |  Klick → fixieren",
        color="#cccccc", fontsize=9, y=0.998)

    # -------------------------------------------------------
    # Vertikale Linien — alle Achsen gleicher x-Raum (0..min_len)
    # -------------------------------------------------------
    chart_axes = [top_ax] + macd_axs  # MACD + Top
    heat_axes  = [heat_ax, comp_ax]   # Heatmaps separat

    start_x = min_len // 2

    vlines_chart = [ax.axvline(x=start_x, color="#ffff00", lw=1.2, alpha=0.9, ls="--")
                    for ax in chart_axes]
    vlines_heat  = [ax.axvline(x=start_x, color="#ffff00", lw=1.2, alpha=0.9, ls="--")
                    for ax in heat_axes]

    all_vlines = vlines_chart + vlines_heat
    all_interactive_axes = chart_axes + heat_axes

    # -------------------------------------------------------
    # Status Panel aufbauen
    # -------------------------------------------------------
    def build_status(xi):
        status_ax.cla()
        status_ax.set_facecolor("#0a0a0a")
        status_ax.set_xlim(0, 1)
        status_ax.set_ylim(0, 1)
        status_ax.axis("off")
        status_ax.set_title("◈ FAIB STATUS", color="#4488ff", fontsize=9,
                             fontweight="bold", pad=8)

        idx = max(0, min(int(xi), min_len - 1))
        status_ax.text(0.5, 0.97, f"Index: {idx}", color="#ffff00",
                       fontsize=8, ha="center", va="top", fontweight="bold")
        status_ax.axhline(0.94, color="#333333", lw=0.8, xmin=0.05, xmax=0.95)

        status_ax.text(0.15, 0.91, "Fraktal",   color="#888888", fontsize=7, va="top")
        status_ax.text(0.50, 0.91, "Dominanz",  color="#888888", fontsize=7, va="top", ha="center")
        status_ax.text(0.85, 0.91, "Exp/Komp",  color="#888888", fontsize=7, va="top", ha="center")
        status_ax.axhline(0.88, color="#333333", lw=0.5, xmin=0.05, xmax=0.95)

        row_h = 0.80 / (n_levels + 1)

        for i, F in enumerate(panel_levels):
            d = data[F]
            y_pos = 0.86 - i * row_h

            dom_val = d["dom"][idx]
            exp_val = d["exp"][idx]

            dom_color = "#1f7a1f" if dom_val == 1 else "#b22222"
            dom_label = "LONG ▲"  if dom_val == 1 else "SHORT ▼"
            exp_color = "#e8a010" if exp_val == 1 else "#1a6faf"
            exp_label = "EXP ↑"   if exp_val == 1 else "COMP ↓"

            status_ax.text(0.12, y_pos, f"F={F}", color="#aaaaaa",
                           fontsize=7, va="center", fontweight="bold")

            dom_box = FancyBboxPatch((0.30, y_pos - 0.014), 0.33, 0.030,
                                     boxstyle="round,pad=0.01",
                                     facecolor=dom_color, alpha=0.9,
                                     transform=status_ax.transAxes)
            status_ax.add_patch(dom_box)
            status_ax.text(0.465, y_pos, dom_label, color="white",
                           fontsize=6.5, va="center", ha="center", fontweight="bold")

            exp_box = FancyBboxPatch((0.66, y_pos - 0.014), 0.30, 0.030,
                                     boxstyle="round,pad=0.01",
                                     facecolor=exp_color, alpha=0.9,
                                     transform=status_ax.transAxes)
            status_ax.add_patch(exp_box)
            status_ax.text(0.81, y_pos, exp_label, color="white",
                           fontsize=6.5, va="center", ha="center", fontweight="bold")

        # MACD Werte
        y_sep = 0.86 - n_levels * row_h - 0.02
        status_ax.axhline(y_sep, color="#333333", lw=0.5, xmin=0.05, xmax=0.95)
        status_ax.text(0.5, y_sep - 0.02, "── MACD Werte ──",
                       color="#555555", fontsize=6, ha="center", va="top")

        for i, F in enumerate(panel_levels):
            d = data[F]
            y_pos = y_sep - 0.06 - i * 0.052
            if y_pos < 0.01:
                break
            mv = d["macd"][idx]
            sv = d["signal"][idx]
            dv = mv - sv
            dc = "#1f7a1f" if dv >= 0 else "#b22222"
            status_ax.text(0.5, y_pos,
                           f"F{F}: M={mv:+.1f}  S={sv:+.1f}",
                           color="#777777", fontsize=5.5, ha="center", va="top")
            status_ax.text(0.5, y_pos - 0.024,
                           f"Δ={dv:+.2f}",
                           color=dc, fontsize=6.5, ha="center", va="top", fontweight="bold")

        fig.canvas.draw_idle()

    # Start
    build_status(start_x)

    # -------------------------------------------------------
    # Interaktivität
    # -------------------------------------------------------
    state = {"fixed": False}

    def on_move(event):
        if state["fixed"]:
            return
        if event.inaxes not in all_interactive_axes:
            return
        xi = event.xdata
        if xi is None:
            return
        xi = max(0, min(xi, min_len - 1))
        for vl in all_vlines:
            vl.set_xdata([xi])
        build_status(xi)

    def on_click(event):
        if event.inaxes not in all_interactive_axes:
            return
        state["fixed"] = not state["fixed"]
        if not state["fixed"]:
            xi = event.xdata
            if xi is not None:
                xi = max(0, min(xi, min_len - 1))
                for vl in all_vlines:
                    vl.set_xdata([xi])
                build_status(xi)

    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_press_event",  on_click)

    plt.show()
    print("[viz_interactive] Fenster geschlossen.")
