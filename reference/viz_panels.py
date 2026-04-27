from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def _safe_sample(df, max_points=1800):
    n = len(df)
    if n <= max_points:
        return df.copy()
    step = max(1, n // max_points)
    idx = np.arange(0, n, step)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return df.iloc[idx].copy()


def plot_panels(name, ur, fractals, nexus_df, fusion_series, verlag_series, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    panel_levels = sorted(fractals.keys())
    if not panel_levels:
        return

    sampled = {}
    index_maps = {}
    min_len = None

    for F in panel_levels:
        df_full = fractals[F].copy().reset_index(drop=True)
        df_sampled = _safe_sample(df_full, max_points=1800)
        sampled[F] = df_sampled
        index_maps[F] = df_sampled.index.to_numpy()
        min_len = len(df_sampled) if min_len is None else min(min_len, len(df_sampled))

    if min_len is None or min_len == 0:
        return

    for F in panel_levels:
        idx = index_maps[F][:min_len]
        sampled[F] = sampled[F].iloc[:min_len].copy().reset_index(drop=True)
        index_maps[F] = idx

    n_macd = len(panel_levels)

    fig = plt.figure(figsize=(15, 3.2 * n_macd + 7.0), constrained_layout=True)
    gs = fig.add_gridspec(
        n_macd + 3, 1,
        height_ratios=[1.15] + [1] * n_macd + [1.05] + [1.05],
        hspace=0.28,
    )

    top_ax = fig.add_subplot(gs[0, 0])
    axes = [fig.add_subplot(gs[i + 1, 0]) for i in range(n_macd)]
    heat_ax = fig.add_subplot(gs[n_macd + 1, 0])
    comp_ax = fig.add_subplot(gs[n_macd + 2, 0])

    # -------------------------------------------------
    # Oberstes Panel: dynamisch alle Fraktalebenen
    # -------------------------------------------------
    color_list = [
        "tab:green", "tab:orange", "tab:purple", "tab:red",
        "tab:brown", "tab:pink", "tab:cyan", "tab:olive",
    ]
    top_colors = {F: color_list[i % len(color_list)] for i, F in enumerate(panel_levels)}

    for F in panel_levels:
        dfF = sampled[F]
        x = np.arange(len(dfF))
        if "cum_height" in dfF.columns:
            top_ax.plot(x, dfF["cum_height"].fillna(0.0).to_numpy(),
                        lw=1.1, alpha=0.9, color=top_colors[F], label=f"cum_height F={F}")
        if "close_F" in dfF.columns:
            top_ax.plot(x, dfF["close_F"].fillna(0.0).to_numpy(),
                        lw=1.0, alpha=0.55, color=top_colors[F], linestyle="--", label=f"close_F F={F}")

    level_str = ",".join(str(F) for F in panel_levels)
    top_ax.set_title(f"Fraktal-Close und cum_height (Referenzebenen F={level_str})")
    top_ax.set_ylabel("Wert")
    top_ax.grid(True, alpha=0.2)
    top_ax.legend(loc="best", fontsize=8, ncol=2)

    # -------------------------------------------------
    # Struktur-MACD-Panels
    # -------------------------------------------------
    for ax, F in zip(axes, panel_levels):
        dfF = sampled[F]
        x = np.arange(len(dfF))
        has_struct_macd = all(col in dfF.columns for col in ["c_macd", "c_signal", "c_hist"])
        if not has_struct_macd:
            ax.text(0.5, 0.5, f"F={F}: keine c_* Daten", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"Struktur-MACD F={F}")
            ax.grid(True, alpha=0.2)
            continue

        hist_vals = dfF["c_hist"].fillna(0.0).to_numpy()
        macd_vals = dfF["c_macd"].fillna(0.0).to_numpy()
        signal_vals = dfF["c_signal"].fillna(0.0).to_numpy()

        pos = np.where(hist_vals >= 0, hist_vals, np.nan)
        neg = np.where(hist_vals < 0, hist_vals, np.nan)

        ax.bar(x, pos, color="darkgreen", alpha=0.35, width=1.0, label="c_hist+")
        ax.bar(x, neg, color="darkred", alpha=0.35, width=1.0, label="c_hist-")
        ax.plot(x, macd_vals, color="tab:blue", lw=1.0, label="c_macd")
        ax.plot(x, signal_vals, color="black", lw=1.0, alpha=0.9, label="c_signal")
        ax.axhline(0, color="black", lw=0.8, alpha=0.6)
        ax.set_title(f"Struktur-MACD auf close_F (on_change), F={F}")
        ax.set_ylabel("MACD")
        ax.grid(True, alpha=0.2)
        ax.legend(loc="best", fontsize=8)

    axes[-1].set_xlabel("Zeit / Index")

    # -------------------------------------------------
    # Binäre Dominanz-Heatmap
    # 1 (grün) = c_macd >= c_signal, 0 (rot) = c_macd < c_signal
    # -------------------------------------------------
    bin_matrix = []
    for F in panel_levels:
        df_full = fractals[F].copy().reset_index(drop=True)
        if "c_macd" in df_full.columns and "c_signal" in df_full.columns:
            c_macd = df_full["c_macd"].fillna(0.0).to_numpy()
            c_signal = df_full["c_signal"].fillna(0.0).to_numpy()
            dom_full = np.where(c_macd >= c_signal, 1, 0)
        else:
            dom_full = np.zeros(len(df_full), dtype=int)
        dom_sampled = dom_full[index_maps[F][:min_len]]
        bin_matrix.append(dom_sampled)

    bin_matrix = np.array(bin_matrix)
    cmap = ListedColormap(["#b22222", "#1f7a1f"])
    heat_ax.imshow(bin_matrix, aspect="auto", interpolation="nearest", cmap=cmap, vmin=0, vmax=1)
    heat_ax.set_title("Binäre Dominanzmatrix (1 = Long, 0 = Short)")
    heat_ax.set_yticks(np.arange(len(panel_levels)))
    heat_ax.set_yticklabels([f"F={F}" for F in panel_levels])
    heat_ax.set_ylabel("Fraktal")
    heat_ax.set_xlabel("Binärcodes")

    if min_len > 1:
        heat_ax.set_xticks(np.arange(0, min_len, max(1, min_len // 12)))
        heat_ax.set_xticklabels([])
    heat_ax.tick_params(axis="x", which="both", length=0, pad=18)

    code_step = max(1, min_len // 10)
    for j in range(0, min_len, code_step):
        code = "".join(str(int(bin_matrix[i, j])) for i in range(len(panel_levels)))
        heat_ax.text(j, len(panel_levels) + 0.55, code, ha="center", va="bottom", fontsize=8, clip_on=False)

    heat_ax.set_xticks(np.arange(-0.5, min_len, 1), minor=True)
    heat_ax.set_yticks(np.arange(-0.5, len(panel_levels), 1), minor=True)
    heat_ax.grid(which="minor", color="white", linestyle="-", linewidth=0.25, alpha=0.35)
    heat_ax.tick_params(which="minor", bottom=False, left=False)

    # -------------------------------------------------
    # Kompression / Expansion Heatmap
    # 1 (orange) = c_macd >= 0, 0 (blau) = c_macd < 0
    # -------------------------------------------------
    comp_matrix = []
    for F in panel_levels:
        df_full = fractals[F].copy().reset_index(drop=True)
        if "c_macd" in df_full.columns:
            c_macd_vals = df_full["c_macd"].fillna(0.0).to_numpy()
            comp_full = np.where(c_macd_vals >= 0, 1, 0)
        else:
            comp_full = np.zeros(len(df_full), dtype=int)
        comp_sampled = comp_full[index_maps[F][:min_len]]
        comp_matrix.append(comp_sampled)

    comp_matrix = np.array(comp_matrix)
    cmap_comp = ListedColormap(["#ffd700", "#1a6faf"])  # 0=Kompression=Gelb, 1=Expansion=Blau
    comp_ax.imshow(comp_matrix, aspect="auto", interpolation="nearest", cmap=cmap_comp, vmin=0, vmax=1)
    comp_ax.set_title("Expansion (blau) / Kompression (gelb) — 1 = c_macd >= 0, 0 = c_macd < 0")
    comp_ax.set_yticks(np.arange(len(panel_levels)))
    comp_ax.set_yticklabels([f"F={F}" for F in panel_levels])
    comp_ax.set_ylabel("Fraktal")
    comp_ax.set_xlabel("Kompressionscodes")

    if min_len > 1:
        comp_ax.set_xticks(np.arange(0, min_len, max(1, min_len // 12)))
        comp_ax.set_xticklabels([])
    comp_ax.tick_params(axis="x", which="both", length=0, pad=18)

    code_step_c = max(1, min_len // 10)
    for j in range(0, min_len, code_step_c):
        code = "".join(str(int(comp_matrix[i, j])) for i in range(len(panel_levels)))
        comp_ax.text(j, len(panel_levels) + 0.55, code, ha="center", va="bottom", fontsize=8, clip_on=False)

    comp_ax.set_xticks(np.arange(-0.5, min_len, 1), minor=True)
    comp_ax.set_yticks(np.arange(-0.5, len(panel_levels), 1), minor=True)
    comp_ax.grid(which="minor", color="white", linestyle="-", linewidth=0.25, alpha=0.35)
    comp_ax.tick_params(which="minor", bottom=False, left=False)

    fig.suptitle(
        f"{name} – Fraktal-Close / cum_height / Struktur-MACD / Long-Short- und Kompressions-Heatmaps",
        fontsize=13
    )
    fig.savefig(out_dir / f"{name}_macd_panels_struct.png", dpi=140, bbox_inches="tight")
    plt.close(fig)