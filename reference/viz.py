import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_sample(df: pd.DataFrame, max_points: int = 2000) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()


def _safe_sample_series(s: pd.Series, max_points: int = 2000) -> pd.Series:
    if len(s) <= max_points:
        return s
    step = max(1, len(s) // max_points)
    return s.iloc[::step].copy()


def plot_urdaten_debug(system_name: str, urdaten: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = _safe_sample(urdaten, max_points=3000)

    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)

    axes[0].plot(df.index, df["value"], color="black", lw=1.0)
    axes[0].set_title(f"{system_name} – Originalserie")
    axes[0].grid(True, alpha=0.2)

    axes[1].plot(df.index, df["step_raw"], color="steelblue", lw=0.8)
    axes[1].axhline(0, color="black", lw=0.8, alpha=0.6)
    axes[1].set_title("step_raw")
    axes[1].grid(True, alpha=0.2)

    axes[2].plot(df.index, df["step_norm"], color="darkorange", lw=0.8)
    axes[2].axhline(0, color="black", lw=0.8, alpha=0.6)
    axes[2].set_title("step_norm")
    axes[2].grid(True, alpha=0.2)

    axes[3].plot(df.index, df["cum_height"], color="darkgreen", lw=1.0)
    axes[3].axhline(0, color="black", lw=0.8, alpha=0.6)
    axes[3].set_title("cum_height (Urdaten-Basisbahn)")
    axes[3].grid(True, alpha=0.2)

    fig.tight_layout()
    fig.savefig(out_dir / f"{system_name}_01_urdaten_debug.png", dpi=150)
    plt.close(fig)


def plot_cumheight_with_fractals(system_name: str,
                                 urdaten: pd.DataFrame,
                                 fractals: dict,
                                 out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    ur = urdaten.copy().reset_index(drop=True)
    df = _safe_sample(ur, max_points=2500)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df["cum_height"], color="black", lw=1.2, label="cum_height")

    colors = ["tab:red", "tab:blue", "tab:green", "tab:purple", "tab:orange", "tab:brown",
              "tab:cyan", "tab:olive", "tab:pink", "tab:gray"]

    # Prüfung: nur Fraktale mit aussagekräftigem MACD anzeigen
    valid_fractals = []
    for F in sorted(fractals.keys()):
        df_check = fractals[F]
        if "c_macd" in df_check.columns:
            macd_range = df_check["c_macd"].abs().max()
            threshold = F * 0.001
            if macd_range < threshold:
                print(f"[viz] F={F} übersprungen — c_macd max={macd_range:.4f} < threshold={threshold:.4f}")
                continue
        if "close_F" in df_check.columns:
            if df_check["close_F"].nunique() <= 1:
                print(f"[viz] F={F} übersprungen — close_F hat nur 1 Wert")
                continue
        valid_fractals.append(F)

    print(f"[viz] Aktive Fraktale im Diagramm: {valid_fractals}")

    for i, F in enumerate(valid_fractals):
        dfF = fractals[F].reset_index(drop=True).reindex(df.index)
        ax.step(df.index, dfF["close_F"], where="post",
                color=colors[i % len(colors)], alpha=0.9, lw=1.0, label=f"close_F{F}")

    ax.set_title(f"{system_name} – cum_height und Fraktal-Closes")
    ax.grid(True, alpha=0.2)
    ax.legend(ncol=3)
    fig.tight_layout()
    fig.savefig(out_dir / f"{system_name}_02_cumheight_fractals.png", dpi=150)
    plt.close(fig)


def plot_fractal_closes_grid(system_name: str,
                             fractals: dict,
                             out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    fractal_levels = sorted(fractals.keys())
    n = len(fractal_levels)
    ncols = 2
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.8 * nrows), sharex=False)
    axes = np.array(axes).reshape(-1)

    for ax, F in zip(axes, fractal_levels):
        dfF = _safe_sample(fractals[F], max_points=1500)
        ax.step(dfF.index, dfF["close_F"], where="post", color="tab:blue", lw=1.0, label=f"close_F{F}")
        ax.plot(dfF.index, dfF["close_F"] + dfF["dot"], color="tab:red", lw=0.9, alpha=0.85, label="close_F + dot")
        ax.set_title(f"Fraktal F={F}")
        ax.grid(True, alpha=0.2)
        ax.legend()

    for ax in axes[len(fractal_levels):]:
        ax.axis("off")

    fig.suptitle(f"{system_name} – Fraktal-Closes je Ebene", y=0.995, fontsize=13)
    fig.tight_layout()
    fig.savefig(out_dir / f"{system_name}_03_fractal_closes_grid.png", dpi=150)
    plt.close(fig)


def plot_nexus_debug(system_name: str,
                     nexus_df: pd.DataFrame,
                     fusion_series: pd.Series,
                     verlag_series: pd.Series,
                     out_dir: Path,
                     fractals: dict = None,
                     urdaten: pd.DataFrame = None,
                     verlag_full: pd.Series = None):
    out_dir.mkdir(parents=True, exist_ok=True)

    nx = nexus_df.reset_index(drop=True).copy()
    fu = fusion_series.reset_index(drop=True).copy()
    ve = verlag_series.reset_index(drop=True).copy()

    nx = _safe_sample(nx, max_points=2500)

    fu = fu.iloc[nx.index]
    ve = ve.iloc[nx.index]

    # Filtern: nur Fraktale mit aussagekräftigem MACD
    valid_cols = []
    if fractals is not None:
        for col in nx.columns:
            F = float(col.replace("F", ""))
            if F in fractals:
                df_check = fractals[F]
                if "c_macd" in df_check.columns:
                    macd_range = df_check["c_macd"].abs().max()
                    threshold = F * 0.001
                    if macd_range < threshold:
                        print(f"[nexus_debug] F={F} übersprungen")
                        continue
            valid_cols.append(col)
    else:
        valid_cols = list(nx.columns)

    nx_filtered = nx[valid_cols]

    # Originalserie oben wenn vorhanden
    has_ur = urdaten is not None and "value" in urdaten.columns
    n_verl = 7  # Anzahl Verlagerungs-Panels
    # 1 cum_height + 1 nexus + 1 fusion + 1 fusion_gewichtet + 7 verlagerungen
    hr_top = [1.8, 1.2, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8] if has_ur else [1.2, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8]
    height_ratios = hr_top + [0.7] * n_verl

    fig, axes = plt.subplots(len(height_ratios), 1,
                             figsize=(14, sum(height_ratios) * 2.2),
                             sharex=False,
                             gridspec_kw={"height_ratios": height_ratios,
                                          "hspace": 0.4})

    x = np.arange(len(nx))
    ax_offset = 0

    # Panel 0: cum_height + close_F — auf gleichem X wie Nexus (nx.index = x)
    if has_ur:
        ur_reset = urdaten.reset_index(drop=True)

        # Auf gleiche Länge wie nx samplen — damit sharex funktioniert
        n_nx = len(nx)
        if len(ur_reset) > n_nx:
            step = max(1, len(ur_reset) // n_nx)
            ur_plot = ur_reset.iloc[::step].iloc[:n_nx].reset_index(drop=True)
        else:
            ur_plot = ur_reset.iloc[:n_nx].reset_index(drop=True)

        axes[0].plot(x, ur_plot["cum_height"].fillna(0.0).to_numpy(),
                     color="black", lw=1.2, label="cum_height")

        fraktal_colors = ["tab:red", "tab:blue", "tab:green", "tab:purple",
                          "tab:orange", "tab:brown", "tab:cyan", "tab:olive"]

        if fractals is not None:
            valid_F_list = [float(c.replace("F","")) for c in valid_cols
                            if float(c.replace("F","")) in fractals]
            for fi, F in enumerate(valid_F_list):
                dfF = fractals[F].reset_index(drop=True)
                if len(dfF) > n_nx:
                    step_f = max(1, len(dfF) // n_nx)
                    dfF_plot = dfF.iloc[::step_f].iloc[:n_nx].reset_index(drop=True)
                else:
                    dfF_plot = dfF.iloc[:n_nx].reset_index(drop=True)
                if "close_F" in dfF_plot.columns:
                    axes[0].step(x, dfF_plot["close_F"].fillna(0.0).to_numpy(),
                                 where="post",
                                 color=fraktal_colors[fi % len(fraktal_colors)],
                                 alpha=0.85, lw=1.0, label=f"close_F{int(F)}")

        axes[0].set_title(f"{system_name} – cum_height + Fraktal-Closes (aktive Ebenen)")
        axes[0].grid(True, alpha=0.2)
        axes[0].legend(ncol=4, fontsize=7)
        ax_offset = 1

    for col in nx_filtered.columns:
        axes[ax_offset].step(x, nx_filtered[col].to_numpy(), where="post", lw=0.8, label=col)
    axes[ax_offset].set_title(f"NEXUS-Matrix (binäre Dominanz je Fraktal) — {len(valid_cols)} aktive Ebenen")
    axes[ax_offset].set_ylim(-0.1, 1.1)
    axes[ax_offset].grid(True, alpha=0.2)
    axes[ax_offset].legend(ncol=min(5, len(valid_cols)))

    # Fusion nur aus aktiven Fraktalen neu berechnen
    fu_dom_list = []
    for col in valid_cols:
        F = float(col.replace("F", ""))
        if F in fractals:
            df_f = fractals[F]
            if "c_macd" in df_f.columns and "c_signal" in df_f.columns:
                dom = (df_f["c_macd"].fillna(0.0) >= df_f["c_signal"].fillna(0.0)).astype(int)
            else:
                dom = pd.Series(np.zeros(len(df_f)))
            fu_dom_list.append(dom.reset_index(drop=True).rename(col))

    if fu_dom_list:
        fu_df = pd.concat(fu_dom_list, axis=1).replace({1:1, 0:-1})
        fu_aktiv = fu_df.sum(axis=1).reset_index(drop=True)
        # Auf nx Länge samplen
        if len(fu_aktiv) > n_nx:
            step_fu = max(1, len(fu_aktiv) // n_nx)
            fu_plot = fu_aktiv.iloc[::step_fu].iloc[:n_nx].reset_index(drop=True)
        else:
            fu_plot = fu_aktiv.iloc[:n_nx].reset_index(drop=True)
    else:
        fu_plot = fu.iloc[:n_nx]

    axes[ax_offset + 1].plot(x[:len(fu_plot)], fu_plot.to_numpy(),
                             color="darkgreen", lw=1.0)
    axes[ax_offset + 1].axhline(0, color="black", lw=0.8, alpha=0.6)
    axes[ax_offset + 1].set_title(f"Fusion — nur aktive Fraktale {valid_cols}")
    axes[ax_offset + 1].grid(True, alpha=0.2)

    # ── Gewichtete Fusion ────────────────────────────────────
    # Größere Fraktale bekommen mehr Gewicht
    # Gewicht i+1 für Fraktal an Position i (aufsteigend sortiert)
    # Summe = 1+2+...+n = n*(n+1)/2
    n_vc_fu = len(fu_dom_list)
    gewichte = list(range(1, n_vc_fu + 1))  # [1,2,3,...,n]
    summe_gew = sum(gewichte)               # n*(n+1)/2

    if fu_dom_list:
        fu_gew_list = []
        for i, dom_s in enumerate(fu_dom_list):
            gew = gewichte[i] / summe_gew
            signed = dom_s.replace({1: 1, 0: -1}) * gew
            fu_gew_list.append(signed)

        fu_gew = pd.concat(fu_gew_list, axis=1).sum(axis=1).reset_index(drop=True)

        if len(fu_gew) > n_nx:
            step_fg = max(1, len(fu_gew) // n_nx)
            fu_gew_plot = fu_gew.iloc[::step_fg].iloc[:n_nx].reset_index(drop=True)
        else:
            fu_gew_plot = fu_gew.iloc[:n_nx].reset_index(drop=True)

        # Gewichtungs-Info für Titel
        gew_info = " | ".join([f"{valid_cols[i]}={gewichte[i]}/{summe_gew}"
                               for i in range(n_vc_fu)])

        axes[ax_offset + 2].plot(x[:len(fu_gew_plot)], fu_gew_plot.to_numpy(),
                                 color="#4488ff", lw=1.0)
        axes[ax_offset + 2].fill_between(x[:len(fu_gew_plot)],
                                          fu_gew_plot.to_numpy(), 0,
                                          where=fu_gew_plot.to_numpy() >= 0,
                                          alpha=0.2, color="darkgreen")
        axes[ax_offset + 2].fill_between(x[:len(fu_gew_plot)],
                                          fu_gew_plot.to_numpy(), 0,
                                          where=fu_gew_plot.to_numpy() < 0,
                                          alpha=0.2, color="darkred")
        axes[ax_offset + 2].axhline(0, color="black", lw=0.8, alpha=0.6)
        axes[ax_offset + 2].set_title(
            "Fusion GEWICHTET — groessere Fraktale dominieren | " + gew_info,
            fontsize=7)
        axes[ax_offset + 2].grid(True, alpha=0.2)

    # ── Fusion GEWICHTET INVERS — kleine Fraktale dominieren ─
    # Kleine Fraktale = explosiv, impulsiv, nichtlinear → mehr Gewicht
    if fu_dom_list:
        gewichte_inv = list(reversed(gewichte))  # [n, n-1, ..., 2, 1]

        fu_gew_inv_list = []
        for i, dom_s in enumerate(fu_dom_list):
            gew = gewichte_inv[i] / summe_gew
            signed = dom_s.replace({1: 1, 0: -1}) * gew
            fu_gew_inv_list.append(signed)

        fu_gew_inv = pd.concat(fu_gew_inv_list, axis=1).sum(axis=1).reset_index(drop=True)

        if len(fu_gew_inv) > n_nx:
            step_fi = max(1, len(fu_gew_inv) // n_nx)
            fu_gew_inv_plot = fu_gew_inv.iloc[::step_fi].iloc[:n_nx].reset_index(drop=True)
        else:
            fu_gew_inv_plot = fu_gew_inv.iloc[:n_nx].reset_index(drop=True)

        gew_inv_info = " | ".join([f"{valid_cols[i]}={gewichte_inv[i]}/{summe_gew}"
                                   for i in range(n_vc_fu)])

        axes[ax_offset + 3].plot(x[:len(fu_gew_inv_plot)], fu_gew_inv_plot.to_numpy(),
                                 color="#ff8800", lw=1.0)
        axes[ax_offset + 3].fill_between(x[:len(fu_gew_inv_plot)],
                                          fu_gew_inv_plot.to_numpy(), 0,
                                          where=fu_gew_inv_plot.to_numpy() >= 0,
                                          alpha=0.2, color="darkgreen")
        axes[ax_offset + 3].fill_between(x[:len(fu_gew_inv_plot)],
                                          fu_gew_inv_plot.to_numpy(), 0,
                                          where=fu_gew_inv_plot.to_numpy() < 0,
                                          alpha=0.2, color="darkred")
        axes[ax_offset + 3].axhline(0, color="black", lw=0.8, alpha=0.6)
        axes[ax_offset + 3].set_title(
            "Fusion GEWICHTET INVERS — kleine Fraktale dominieren (explosiv/impulsiv) | " + gew_inv_info,
            fontsize=7)
        axes[ax_offset + 3].grid(True, alpha=0.2)

    # ── Fusion 4-7: Kombinationen aus Fusion 2 + 3 ──────────
    if fu_dom_list:
        # Auf gleiche Länge bringen
        n_min = min(len(fu_gew_plot), len(fu_gew_inv_plot))
        f2 = fu_gew_plot.to_numpy()[:n_min]
        f3 = fu_gew_inv_plot.to_numpy()[:n_min]
        x4 = x[:n_min]

        # A: Differenz F3 - F2 (Spannung Impuls vs Struktur)
        fusion_diff = f3 - f2

        # B: Produkt F2 * F3 (Konfluenz)
        fusion_prod = f2 * f3

        # C: Normierte Summe (Balance)
        fusion_balance = (f2 + f3) / 2.0

        # D: Verhältnis F3 / (|F2| + epsilon)
        fusion_ratio = f3 / (np.abs(f2) + 0.001)
        # Begrenzen auf sinnvollen Bereich
        fusion_ratio = np.clip(fusion_ratio, -5, 5)

        kombinationen = [
            (fusion_diff,    "Fusion A — Differenz (F3-F2): Spannung Impuls vs Struktur",    "#00ffcc", "#ff00cc"),
            (fusion_prod,    "Fusion B — Produkt (F2*F3): Konfluenz (+ aligned, - gegenl.)", "#ffcc00", "#cc00ff"),
            (fusion_balance, "Fusion C — Balance (F2+F3)/2: Ausgewogen",                     "#88ff88", "#ff8888"),
            (fusion_ratio,   "Fusion D — Ratio (F3/|F2|): Impuls relativ zu Struktur",       "#ff8800", "#0088ff"),
        ]

        for i, (fus_arr, titel, cp, cn) in enumerate(kombinationen):
            ax_idx = ax_offset + 4 + i
            axes[ax_idx].plot(x4, fus_arr, color=cp, lw=0.8, alpha=0.9)
            axes[ax_idx].fill_between(x4, fus_arr, 0,
                                       where=fus_arr >= 0, alpha=0.2, color=cp)
            axes[ax_idx].fill_between(x4, fus_arr, 0,
                                       where=fus_arr < 0, alpha=0.2, color=cn)
            axes[ax_idx].axhline(0, color="black", lw=0.8, alpha=0.6)
            axes[ax_idx].set_title(titel, fontsize=7)
            axes[ax_idx].grid(True, alpha=0.2)

    # ── Verlagerungs-Subsets berechnen ──────────────────────
    def build_verl_subset(cols_subset):
        dl = []
        for col in cols_subset:
            F = float(col.replace("F",""))
            if F in fractals:
                df_f = fractals[F]
                if "c_macd" in df_f.columns and "c_signal" in df_f.columns:
                    dom = (df_f["c_macd"].fillna(0.0) >= df_f["c_signal"].fillna(0.0)).astype(int)
                else:
                    dom = pd.Series(np.zeros(len(df_f)))
                dl.append(dom.reset_index(drop=True).rename(col))
        if not dl:
            return pd.Series(np.zeros(10))
        dom_df = pd.concat(dl, axis=1).replace({1:1, 0:-1})
        return dom_df.sum(axis=1).diff().fillna(0.0).reset_index(drop=True)

    # Subsets
    n_vc  = len(valid_cols)
    mid   = n_vc // 2
    m1    = max(0, mid - 1)
    m2    = min(n_vc - 1, mid + 1)

    verl_alle      = build_verl_subset(valid_cols)
    verl_mitte_3   = build_verl_subset(valid_cols[m1:m2+1])
    verl_mitte_1   = build_verl_subset([valid_cols[mid]])
    verl_klein      = build_verl_subset(valid_cols[:mid])
    verl_gross      = build_verl_subset(valid_cols[mid:])
    verl_groesster  = build_verl_subset([valid_cols[-1]])
    verl_kleinster  = build_verl_subset([valid_cols[0]])

    verlagerungen = [
        (verl_alle,    f"Verlagerung ALLE ({n_vc} Fraktale)",         "darkgreen",  "darkred"),
        (verl_mitte_3, f"Verlagerung MITTE ±1 — {valid_cols[m1:m2+1]}", "#ccaa00", "#cc4400"),
        (verl_mitte_1, f"Verlagerung MITTE — {valid_cols[mid]}",      "#aaaaaa",   "#ff4444"),
        (verl_klein,   f"Verlagerung KLEIN — {valid_cols[:mid]}",     "#4488ff",   "#b22222"),
        (verl_gross,   f"Verlagerung GROSS — {valid_cols[mid:]}",     "#e8a010",   "#b22222"),
        (verl_groesster, f"Verlagerung GRÖSSTER — {valid_cols[-1]}",  "#ff4444",   "#4488ff"),
        (verl_kleinster, f"Verlagerung KLEINSTER — {valid_cols[0]}", "#4488ff",   "#ff4444"),
    ]

    def draw_verl(ax, verl, title, cp, cn):
        v = verl.to_numpy()
        n_ve = len(v)
        x_scale = n_nx / n_ve if n_ve > 0 else 1
        nz_idx = np.where(v != 0)[0]
        nz_x = nz_idx * x_scale
        nz_v = v[nz_idx]
        nz_c = [cp if val >= 0 else cn for val in nz_v]
        if len(nz_idx) > 0:
            ax.bar(nz_x, nz_v, color=nz_c, alpha=0.85, width=max(1, x_scale*2))
        ax.set_xlim(0, n_nx)
        ax.axhline(0, color="black", lw=0.8, alpha=0.6)
        ax.set_title(f"{title} — {len(nz_idx)} Signale", fontsize=8)
        ax.grid(True, alpha=0.2)

    for i, (verl, title, cp, cn) in enumerate(verlagerungen):
        draw_verl(axes[ax_offset + 8 + i], verl, title, cp, cn)

    axes[ax_offset + 8 + len(verlagerungen) - 1].set_xlabel("Tick-Index")

    fig.tight_layout()
    fig.savefig(out_dir / f"{system_name}_04_nexus_debug.png", dpi=150)
    plt.close(fig)


def plot_single_fractal_chart(system_name: str,
                              urdaten: pd.DataFrame,
                              F: float,
                              fractal_df: pd.DataFrame,
                              out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    df = fractal_df.copy()
    df["active_level"] = df["close_F"] + df["dot"]

    ur = urdaten.reindex(df.index)
    df["step_dir"] = np.sign(ur["step_norm"].fillna(0.0))

    plot_df = _safe_sample(df, max_points=1800).copy()
    x = np.arange(len(plot_df))

    # MACD auf Fraktalzustand (s_*)
    has_macd_state = all(col in plot_df.columns for col in ["s_macd", "s_signal", "s_hist"])

    if has_macd_state:
        fig, axes = plt.subplots(
            3, 1, figsize=(15, 11), sharex=True,
            gridspec_kw={"height_ratios": [3.5, 1.2, 1.6]}
        )
        ax = axes[0]
        ax2 = axes[1]
        ax3 = axes[2]
    else:
        fig, axes = plt.subplots(
            2, 1, figsize=(15, 9), sharex=True,
            gridspec_kw={"height_ratios": [3.5, 1.4]}
        )
        ax = axes[0]
        ax2 = axes[1]
        ax3 = None

    # Oberes Panel: Fraktal-Chart
    y_min = float(np.nanmin(np.minimum(plot_df["close_F"], plot_df["active_level"])))
    y_max = float(np.nanmax(np.maximum(plot_df["close_F"], plot_df["active_level"])))
    start_grid = math.floor(y_min / F) * F
    end_grid = math.ceil(y_max / F) * F

    for y in np.arange(start_grid, end_grid + F, F):
        ax.axhline(y, color="lightgray", lw=0.6, alpha=0.6, zorder=0)

    for i, (_, row) in enumerate(plot_df.iterrows()):
        color = "green" if row["step_dir"] >= 0 else "red"
        ax.vlines(i, row["close_F"], row["active_level"], color=color, lw=1.0, alpha=0.85)

    ax.step(x, plot_df["close_F"].to_numpy(), where="post",
            color="black", lw=1.2, label="close_F")

    ax.plot(x, plot_df["active_level"].to_numpy(),
            color="tab:blue", lw=1.0, alpha=0.9, label="close_F + dot")

    ax.set_title(f"{system_name} – Fraktal F={F} (stabiler Close + DOT)")
    ax.set_ylabel("Fraktalhöhe")
    ax.grid(True, alpha=0.15)
    ax.legend()

    # Mittleres Panel: DOT + optional dot_smooth
    ax2.plot(x, plot_df["dot"].to_numpy(), color="tab:orange", lw=1.0, label="dot")
    if "dot_smooth" in plot_df.columns:
        ax2.plot(x, plot_df["dot_smooth"].to_numpy(), color="tab:purple", lw=1.0, alpha=0.9, label="dot_smooth")
    ax2.axhline(0, color="black", lw=0.8, alpha=0.6)
    ax2.axhline(F, color="gray", lw=0.8, ls="--", alpha=0.5, label="Fraktalgrenze +F")
    ax2.axhline(-F, color="gray", lw=0.8, ls="--", alpha=0.5, label="Fraktalgrenze -F")
    ax2.set_title(f"DOT-Verlauf innerhalb Fraktal F={F}")
    ax2.set_ylabel("dot")
    ax2.grid(True, alpha=0.2)
    ax2.legend()

    # Unteres Panel: STRUKTUR-MACD auf close_F (eventbasiert, c_*)
    has_macd_struct = all(col in plot_df.columns for col in ["c_macd", "c_signal", "c_hist"])

    if has_macd_struct and ax3 is not None:
        hist_vals = plot_df["c_hist"].to_numpy()
        hist_colors = ["darkgreen" if h >= 0 else "darkred" for h in hist_vals]

        ax3.bar(x, hist_vals, color=hist_colors, alpha=0.45, width=0.8, label="c_hist")
        ax3.plot(x, plot_df["c_macd"].to_numpy(), color="tab:blue", lw=1.1, label="c_macd")
        ax3.plot(x, plot_df["c_signal"].to_numpy(), color="black", lw=1.0, label="c_signal")
        ax3.axhline(0, color="black", lw=0.8, alpha=0.6)

        ax3.set_title(f"Struktur-MACD auf close_F (on_change), F={F}")
        ax3.set_ylabel("MACD")
        ax3.set_xlabel("Zeit / Index")
        ax3.grid(True, alpha=0.2)
        ax3.legend()
    else:
        ax2.set_xlabel("Zeit / Index")

    fig.tight_layout()
    fig.savefig(out_dir / f"{system_name}_fraktal_F{int(F)}_chart.png", dpi=160)
    plt.close(fig)


def plot_all_single_fractals(system_name: str,
                             urdaten: pd.DataFrame,
                             fractals: dict,
                             out_dir: Path):
    for F in sorted(fractals.keys()):
        plot_single_fractal_chart(system_name, urdaten, F, fractals[F], out_dir)


def plot_nexus(system_name: str,
               urdaten: pd.DataFrame,
               fractals: dict,
               nexus_df: pd.DataFrame,
               fusion_series: pd.Series,
               verlag_series: pd.Series,
               out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_urdaten_debug(system_name, urdaten, out_dir)
    plot_cumheight_with_fractals(system_name, urdaten, fractals, out_dir)
    plot_fractal_closes_grid(system_name, fractals, out_dir)
    plot_nexus_debug(system_name, nexus_df, fusion_series, verlag_series, out_dir,
                     fractals=fractals, urdaten=urdaten,
                     verlag_full=verlag_series)
    plot_all_single_fractals(system_name, urdaten, fractals, out_dir)