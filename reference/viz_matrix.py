"""
FAIB NEXUS — viz_matrix.py
===========================
Bias-Energy Matrix: 2^4 x 2^4 Zustandsmatrix

Y-Achse: große Fraktale (obere Hälfte der aktiven)
X-Achse: kleine Fraktale (untere Hälfte der aktiven)

Zellwert: gewichtete Häufigkeit des Long-Zustands
          aus historischen FAIB-Daten

Aktivierung in main.py:
    ENABLE_VIZ_MATRIX = True

    try:
        import viz_matrix
    except ImportError:
        viz_matrix = None

    if ENABLE_VIZ_MATRIX and viz_matrix is not None:
        viz_matrix.plot_matrix(name, ur, fractals, nexus_df, out_dir)
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from itertools import product


def _get_valid_fractals(fractals: dict) -> list:
    """Gefilterte Fraktale — gleiche Logik wie viz_panels."""
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


def _get_dominance(fractals: dict, F: float) -> np.ndarray:
    """Binäre Dominanz: 1=Long, 0=Short."""
    df = fractals[F]
    if "c_macd" in df.columns and "c_signal" in df.columns:
        dom = (df["c_macd"].fillna(0.0) >= df["c_signal"].fillna(0.0)).astype(int)
    else:
        dom = pd.Series(np.zeros(len(df), dtype=int))
    return dom.reset_index(drop=True).to_numpy()


def _state_to_label(state: tuple) -> str:
    """(1,0,1,1) → 'LSLL'"""
    return "".join("L" if s == 1 else "S" for s in state)


def _all_states(n: int):
    """Alle 2^n Zustände als Tupel."""
    return list(product([0, 1], repeat=n))


def _gewichtung(state: tuple, gewichte: list) -> float:
    """
    Gewichtete Summe des Zustands.
    Ergebnis: -1.0 bis +1.0
    L=+1, S=-1 → Long dominiert = positiv, Short = negativ
    """
    summe = sum(gewichte)
    # L=1 → +1, S=0 → -1 (Dualismus)
    wert = sum((2*s - 1) * w for s, w in zip(state, gewichte))
    return wert / summe if summe > 0 else 0.0


def plot_matrix(name, ur, fractals, nexus_df, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Aktive Fraktale ─────────────────────────────────────
    valid_F = _get_valid_fractals(fractals)
    n_valid = len(valid_F)

    if n_valid < 2:
        print("[viz_matrix] Zu wenige aktive Fraktale!")
        return

    # Aufteilen: untere Hälfte = klein, obere = groß
    half = n_valid // 2
    F_klein = valid_F[:half]   # X-Achse
    F_gross = valid_F[half:]   # Y-Achse

    # Falls ungerade — kleinsten weglassen (bei Y) oder größten (bei X)
    # Ziel: gleiche Länge für quadratische Matrix
    n_dim = min(len(F_klein), len(F_gross))
    F_klein = F_klein[:n_dim]
    F_gross = F_gross[:n_dim]

    print(f"[viz_matrix] Klein (X): {F_klein}")
    print(f"[viz_matrix] Groß  (Y): {F_gross}")
    print(f"[viz_matrix] Matrix: {2**n_dim} x {2**n_dim} = {4**n_dim} Zellen")

    # ── Dominanz-Arrays ─────────────────────────────────────
    dom_klein = {F: _get_dominance(fractals, F) for F in F_klein}
    dom_gross = {F: _get_dominance(fractals, F) for F in F_gross}

    # Auf gleiche Länge bringen
    n_ref = min(len(dom_klein[F_klein[0]]), len(dom_gross[F_gross[0]]))
    for F in F_klein:
        dom_klein[F] = dom_klein[F][:n_ref]
    for F in F_gross:
        dom_gross[F] = dom_gross[F][:n_ref]

    # ── Alle möglichen Zustände ──────────────────────────────
    # Umgekehrt: LLLL zuerst → Long oben links, Short unten rechts
    states = list(reversed(_all_states(n_dim)))
    n_states = len(states)  # 2^n_dim

    # Gewichtung: Position 0 = kleinstes Gewicht, letztes = größtes
    gewichte_klein = list(range(1, n_dim + 1))  # [1,2,...,n]
    gewichte_gross = list(range(1, n_dim + 1))

    # ── Matrix 1: Theoretische Gewichtung ───────────────────
    # Zellwert = Durchschnitt der Gewichtungen von Y und X Zustand
    theo_matrix = np.zeros((n_states, n_states))
    for yi, y_state in enumerate(states):
        for xi, x_state in enumerate(states):
            gw_y = _gewichtung(y_state, gewichte_gross)  # -1 bis +1
            gw_x = _gewichtung(x_state, gewichte_klein)  # -1 bis +1
            theo_matrix[yi, xi] = (gw_y + gw_x) / 2.0   # -1 bis +1

    # ── Matrix 2: Historische Häufigkeit aus Daten ───────────
    hist_matrix = np.zeros((n_states, n_states))
    count_matrix = np.zeros((n_states, n_states), dtype=int)

    # State-Index Lookup
    state_to_idx = {s: i for i, s in enumerate(states)}

    # Für jeden Tick: Zustand bestimmen
    for t in range(n_ref):
        x_state = tuple(int(dom_klein[F][t]) for F in F_klein)
        y_state = tuple(int(dom_gross[F][t]) for F in F_gross)

        xi = state_to_idx.get(x_state, -1)
        yi = state_to_idx.get(y_state, -1)

        if xi >= 0 and yi >= 0:
            # Dualismus: L=+1, S=-1
            # Durchschnitt aller Fraktale → -1 bis +1
            signed_x = sum(2*s - 1 for s in x_state) / n_dim
            signed_y = sum(2*s - 1 for s in y_state) / n_dim
            wert = (signed_x + signed_y) / 2.0
            hist_matrix[yi, xi] += wert
            count_matrix[yi, xi] += 1

    # Normieren: Durchschnitt pro Zelle
    with np.errstate(divide='ignore', invalid='ignore'):
        hist_norm = np.where(count_matrix > 0,
                             hist_matrix / count_matrix,
                             np.nan)

    # ── Matrix 3: Differenz (live - theo) ───────────────────
    diff_matrix = np.where(~np.isnan(hist_norm),
                           hist_norm - theo_matrix,
                           np.nan)

    # ── Labels ──────────────────────────────────────────────
    labels = [_state_to_label(s) for s in states]

    # ── Plot ────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(24, 8),
                              facecolor="#0d0d0d")

    cmap_rg  = mcolors.LinearSegmentedColormap.from_list(
        "rg", ["#b22222", "#ffff88", "#1f7a1f"])
    cmap_div = mcolors.LinearSegmentedColormap.from_list(
        "div", ["#1a6faf", "#ffffff", "#b22222"])

    def draw_matrix(ax, mat, title, cmap, vmin, vmax, fmt=".0%"):
        ax.set_facecolor("#111111")
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax,
                       aspect="auto", interpolation="nearest")

        # Zellwerte
        for yi in range(n_states):
            for xi in range(n_states):
                val = mat[yi, xi]
                if np.isnan(val):
                    txt = "—"
                    col = "#444444"
                elif fmt == "+.0%":
                    txt = f"{val:+.0%}"
                    col = "white" if abs(val) > 0.3 else "#333333"
                elif fmt == ".0%":
                    txt = f"{val:.0%}"
                    col = "white" if abs(val - 0.5) > 0.2 else "#333333"
                else:
                    txt = f"{val:+.2f}"
                    col = "white" if abs(val) > 0.1 else "#333333"
                ax.text(xi, yi, txt, ha="center", va="center",
                        fontsize=5.5, color=col, fontweight="bold")

        ax.set_xticks(range(n_states))
        ax.set_yticks(range(n_states))
        ax.set_xticklabels(labels, rotation=90, fontsize=6, color="#aaaaaa")
        ax.set_yticklabels(labels, fontsize=6, color="#aaaaaa")
        ax.set_xlabel(f"Klein: {[int(f) for f in F_klein]}",
                      color="#888888", fontsize=8)
        ax.set_ylabel(f"Groß: {[int(f) for f in F_gross]}",
                      color="#888888", fontsize=8)
        ax.set_title(title, color="#cccccc", fontsize=9, pad=8)
        ax.tick_params(colors="#555555")

        # Aktueller Zustand — transparenter Kreis (Hintergrund sichtbar)
        x_curr = tuple(int(dom_klein[F][-1]) for F in F_klein)
        y_curr = tuple(int(dom_gross[F][-1]) for F in F_gross)
        xi_curr = state_to_idx.get(x_curr, -1)
        yi_curr = state_to_idx.get(y_curr, -1)
        if xi_curr >= 0 and yi_curr >= 0:
            # Transparenter Kreis — Zellwert bleibt sichtbar
            ax.plot(xi_curr, yi_curr, "o",
                    color="none",
                    markersize=18,
                    markeredgecolor="#ffff00",
                    markeredgewidth=2.5,
                    zorder=10,
                    label=f"Aktuell: {_state_to_label(y_curr)}/{_state_to_label(x_curr)}")
            ax.legend(loc="upper right", fontsize=6,
                      facecolor="#1a1a1a", edgecolor="#333333",
                      labelcolor="#ffff00")

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(
            colors="#888888", labelsize=6)

    # Matrix 1: Theoretisch
    draw_matrix(axes[0], theo_matrix,
                "Matrix 1 — Theoretische Gewichtung\n(+100%=LLLL, -100%=SSSS)",
                cmap_rg, -1, 1, fmt="+.0%")

    # Matrix 2: Historisch
    draw_matrix(axes[1], hist_norm,
                f"Matrix 2 — Historische Häufigkeit\n({n_ref:,} Ticks | {name})",
                cmap_rg, -1, 1, fmt="+.0%")

    # Matrix 3: Differenz
    draw_matrix(axes[2], diff_matrix,
                "Matrix 3 — Abweichung (Historisch - Theorie)\nblau=unter Erwartung, rot=über Erwartung",
                cmap_div, -0.5, 0.5, fmt="+.2f")

    fig.suptitle(
        f"FAIB Bias-Energy Matrix — {name}\n"
        f"Y={[int(f) for f in F_gross]} (groß) | X={[int(f) for f in F_klein]} (klein) | "
        f"O = aktueller Zustand",
        color="#cccccc", fontsize=11, y=1.01
    )

    plt.tight_layout()
    out_path = out_dir / f"{name}_bias_energy_matrix.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor="#0d0d0d")
    plt.close(fig)
    print(f"[viz_matrix] Gespeichert: {out_path}")

    # ── Matrix 4: Bestätigte Zustände ──────────────────────
    # Nur wo Matrix 1 UND Matrix 3 in gleicher Richtung zeigen
    confirmed_matrix = np.full((n_states, n_states), np.nan)
    confirmed_entries = []  # für Tabelle 5

    for yi in range(n_states):
        for xi in range(n_states):
            theo_val = theo_matrix[yi, xi]
            diff_val = diff_matrix[yi, xi]
            hist_val = hist_norm[yi, xi]

            if np.isnan(hist_val):
                continue

            # Bestätigung: Theorie und Abweichung zeigen gleiche Richtung
            if theo_val > 0 and diff_val > 0:
                confirmed_matrix[yi, xi] = hist_val
                confirmed_entries.append({
                    "y_state": states[yi],
                    "x_state": states[xi],
                    "theo": theo_val,
                    "hist": hist_val,
                    "diff": diff_val,
                    "direction": "LONG",
                })
            elif theo_val < 0 and diff_val < 0:
                confirmed_matrix[yi, xi] = hist_val
                confirmed_entries.append({
                    "y_state": states[yi],
                    "x_state": states[xi],
                    "theo": theo_val,
                    "hist": hist_val,
                    "diff": diff_val,
                    "direction": "SHORT",
                })

    print(f"[viz_matrix] Bestaetigte Zustaende: {len(confirmed_entries)}")

    # ── Instabile Zustaende (Widerspruch) ────────────────────
    unstable_entries = []
    unstable_matrix = np.full((n_states, n_states), np.nan)

    for yi in range(n_states):
        for xi in range(n_states):
            theo_val = theo_matrix[yi, xi]
            diff_val = diff_matrix[yi, xi]
            hist_val = hist_norm[yi, xi]
            if np.isnan(hist_val):
                continue
            if theo_val > 0 and diff_val < 0:
                unstable_matrix[yi, xi] = hist_val
                unstable_entries.append({
                    "y_state": states[yi], "x_state": states[xi],
                    "theo": theo_val, "hist": hist_val,
                    "diff": diff_val, "direction": "LONG_UNSTABLE",
                })
            elif theo_val < 0 and diff_val > 0:
                unstable_matrix[yi, xi] = hist_val
                unstable_entries.append({
                    "y_state": states[yi], "x_state": states[xi],
                    "theo": theo_val, "hist": hist_val,
                    "diff": diff_val, "direction": "SHORT_UNSTABLE",
                })

    print(f"[viz_matrix] Instabile Zustaende: {len(unstable_entries)}")

    # ── Plot Matrix 4 ────────────────────────────────────────
    fig4, (ax4, ax4b) = plt.subplots(1, 2, figsize=(22, 9), facecolor="#0d0d0d")
    ax4.set_facecolor("#111111")
    ax4b.set_facecolor("#111111")

    im4 = ax4.imshow(confirmed_matrix, cmap=cmap_rg,
                     vmin=-1, vmax=1, aspect="auto",
                     interpolation="nearest")

    for yi in range(n_states):
        for xi in range(n_states):
            val = confirmed_matrix[yi, xi]
            if np.isnan(val):
                ax4.text(xi, yi, "—", ha="center", va="center",
                         fontsize=5, color="#333333")
            else:
                col = "white" if abs(val) > 0.3 else "#333333"
                ax4.text(xi, yi, f"{val:+.0%}", ha="center", va="center",
                         fontsize=5.5, color=col, fontweight="bold")

    ax4.set_xticks(range(n_states))
    ax4.set_yticks(range(n_states))
    ax4.set_xticklabels(labels, rotation=90, fontsize=6, color="#aaaaaa")
    ax4.set_yticklabels(labels, fontsize=6, color="#aaaaaa")
    ax4.set_xlabel(f"Klein: {[int(f) for f in F_klein]}", color="#888888", fontsize=8)
    ax4.set_ylabel(f"Groß: {[int(f) for f in F_gross]}", color="#888888", fontsize=8)
    ax4.set_title(
        "Matrix 4 — Bestaetigte Zustaende\n(Theorie + Abweichung zeigen gleiche Richtung)",
        color="#cccccc", fontsize=9)

    # Aktueller Zustand
    x_curr = tuple(int(dom_klein[F][-1]) for F in F_klein)
    y_curr = tuple(int(dom_gross[F][-1]) for F in F_gross)
    xi_curr = state_to_idx.get(x_curr, -1)
    yi_curr = state_to_idx.get(y_curr, -1)
    if xi_curr >= 0 and yi_curr >= 0:
        ax4.plot(xi_curr, yi_curr, "o", color="none", markersize=18,
                 markeredgecolor="#ffff00", markeredgewidth=2.5, zorder=10)

    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04).ax.tick_params(
        colors="#888888", labelsize=6)

    # Matrix 4b — Instabil
    im4b = ax4b.imshow(unstable_matrix, cmap=cmap_div,
                       vmin=-1, vmax=1, aspect="auto", interpolation="nearest")
    for yi in range(n_states):
        for xi in range(n_states):
            val = unstable_matrix[yi, xi]
            if np.isnan(val):
                ax4b.text(xi, yi, "—", ha="center", va="center",
                          fontsize=5, color="#333333")
            else:
                col = "white" if abs(val) > 0.3 else "#333333"
                ax4b.text(xi, yi, f"{val:+.0%}", ha="center", va="center",
                          fontsize=5.5, color=col, fontweight="bold")
    ax4b.set_xticks(range(n_states))
    ax4b.set_yticks(range(n_states))
    ax4b.set_xticklabels(labels, rotation=90, fontsize=6, color="#aaaaaa")
    ax4b.set_yticklabels(labels, fontsize=6, color="#aaaaaa")
    ax4b.set_xlabel(f"Klein: {[int(f) for f in F_klein]}", color="#888888", fontsize=8)
    ax4b.set_ylabel(f"Gross: {[int(f) for f in F_gross]}", color="#888888", fontsize=8)
    ax4b.set_title("Matrix 4b - INSTABILE Zustaende | Widerspruch Theorie vs Abweichung",
                   color="#cccccc", fontsize=9)
    if xi_curr >= 0 and yi_curr >= 0:
        ax4b.plot(xi_curr, yi_curr, "o", color="none", markersize=18,
                  markeredgecolor="#ffff00", markeredgewidth=2.5, zorder=10)
    plt.colorbar(im4b, ax=ax4b, fraction=0.046, pad=0.04).ax.tick_params(
        colors="#888888", labelsize=6)

    fig4.suptitle(f"Matrix 4 — Stabile vs Instabile Zustaende | {name}",
                  color="#cccccc", fontsize=11)
    plt.tight_layout()
    out4 = out_dir / f"{name}_matrix4_stabil_instabil.png"
    fig4.savefig(out4, dpi=130, bbox_inches="tight", facecolor="#0d0d0d")
    plt.close(fig4)
    print(f"[viz_matrix] Matrix 4a+4b gespeichert: {out4}")

    # ── Tabellen 5a/5b (stabil) + 6a/6b (instabil) ──────────
    if confirmed_entries or unstable_entries:
        confirmed_entries.sort(key=lambda e: abs(e["hist"]), reverse=True)
        unstable_entries.sort(key=lambda e: abs(e["hist"]), reverse=True)

        stable_long  = [e for e in confirmed_entries if e["direction"] == "LONG"]
        stable_short = [e for e in confirmed_entries if e["direction"] == "SHORT"]
        unstab_long  = [e for e in unstable_entries if e["direction"] == "LONG_UNSTABLE"]
        unstab_short = [e for e in unstable_entries if e["direction"] == "SHORT_UNSTABLE"]

        print(f"[viz_matrix] Stabil  Long={len(stable_long)} Short={len(stable_short)}")
        print(f"[viz_matrix] Instabil Long={len(unstab_long)} Short={len(unstab_short)}")

        all_F_ordered = list(reversed(F_gross)) + list(reversed(F_klein))
        col_labels = [f"F={int(f)}" for f in all_F_ordered]
        sep_col = len(F_gross)

        def build_table_ax(ax, entries, titel, header_color="#1a1a2e"):
            """Zeichnet eine Tabelle in einen bestehenden Axes."""
            ax.axis("off")
            if not entries:
                ax.text(0.5, 0.5, "Keine Eintraege", ha="center", va="center",
                        transform=ax.transAxes, color="#666666", fontsize=10)
                ax.set_title(titel, color="#cccccc", fontsize=8, pad=6)
                return
            n_r = len(entries)
            tdata, tcolors = [], []
            for e in entries:
                y_bits = list(reversed(e["y_state"]))
                x_bits = list(reversed(e["x_state"]))
                row, rowc = [], []
                for bit in y_bits + x_bits:
                    lbl = "L" if bit == 1 else "S"
                    row.append(lbl)
                    rowc.append("#1f5a1f" if bit == 1 else "#5a1f1f")
                tdata.append(row)
                tcolors.append(rowc)
            tbl = ax.table(cellText=tdata, colLabels=col_labels,
                           cellLoc="center", loc="center", bbox=[0,0,1,0.97])
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(6)
            for j in range(len(col_labels)):
                c = tbl[0, j]
                c.set_facecolor(header_color)
                c.set_text_props(color="#aaaaff", fontweight="bold", fontsize=6)
            for i, (row_data, rowc) in enumerate(zip(tdata, tcolors)):
                for j, (lbl, bg) in enumerate(zip(row_data, rowc)):
                    cell = tbl[i+1, j]
                    cell.set_facecolor(bg)
                    cell.set_text_props(
                        color="#1f7a1f" if lbl == "L" else "#b22222",
                        fontweight="bold")
            for i in range(n_r + 1):
                try:
                    tbl[i, sep_col-1].set_edgecolor("#ffff00")
                    tbl[i, sep_col].set_edgecolor("#ffff00")
                except Exception:
                    pass
            ax.set_title(f"{titel} ({n_r})", color="#cccccc", fontsize=8, pad=6)

        def build_table(entries, titel, filename, header_color="#1a1a2e"):
            if not entries:
                print(f"[viz_matrix] Keine Eintraege fuer: {titel}")
                return
            n_r = len(entries)
            fig_t, ax_t = plt.subplots(
                figsize=(14, max(4, n_r * 0.32 + 2)), facecolor="#0d0d0d")
            ax_t.set_facecolor("#0d0d0d")
            ax_t.axis("off")
            tdata, tcolors = [], []
            for e in entries:
                y_bits = list(reversed(e["y_state"]))
                x_bits = list(reversed(e["x_state"]))
                row, rowc = [], []
                for bit in y_bits + x_bits:
                    lbl = "L" if bit == 1 else "S"
                    row.append(lbl)
                    rowc.append("#1f5a1f" if bit == 1 else "#5a1f1f")
                tdata.append(row)
                tcolors.append(rowc)
            tbl = ax_t.table(cellText=tdata, colLabels=col_labels,
                             cellLoc="center", loc="center", bbox=[0,0,1,1])
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(7)
            for j in range(len(col_labels)):
                c = tbl[0, j]
                c.set_facecolor(header_color)
                c.set_text_props(color="#aaaaff", fontweight="bold")
            for i, (row_data, rowc) in enumerate(zip(tdata, tcolors)):
                for j, (lbl, bg) in enumerate(zip(row_data, rowc)):
                    cell = tbl[i+1, j]
                    cell.set_facecolor(bg)
                    cell.set_text_props(
                        color="#1f7a1f" if lbl == "L" else "#b22222",
                        fontweight="bold")
            for i in range(n_r + 1):
                try:
                    tbl[i, sep_col-1].set_edgecolor("#ffff00")
                    tbl[i, sep_col].set_edgecolor("#ffff00")
                except Exception:
                    pass
            ax_t.set_title(titel, color="#cccccc", fontsize=9, pad=10)
            plt.tight_layout()
            fig_t.savefig(filename, dpi=130, bbox_inches="tight", facecolor="#0d0d0d")
            plt.close(fig_t)
            print(f"[viz_matrix] Gespeichert: {filename}")

        # Alle 4 Tabellen in einem Plot — 2x2 Layout
        max_rows = max(len(stable_long), len(stable_short),
                       len(unstab_long), len(unstab_short), 1)
        fig_all, axes_all = plt.subplots(
            2, 2,
            figsize=(22, max(12, max_rows * 0.28 + 3)),
            facecolor="#0d0d0d")

        build_table_ax(axes_all[0,0], stable_long,
                       "5a — STABIL LONG | Theorie+Abweichung bestaetigt", "#1a2e1a")
        build_table_ax(axes_all[0,1], stable_short,
                       "5b — STABIL SHORT | Theorie+Abweichung bestaetigt", "#2e1a1a")
        build_table_ax(axes_all[1,0], unstab_long,
                       "6a — INSTABIL LONG | Widerspruch Theorie vs Markt", "#1a1a2e")
        build_table_ax(axes_all[1,1], unstab_short,
                       "6b — INSTABIL SHORT | Widerspruch Theorie vs Markt", "#2e2e1a")


        titel_all = ("Fraktal-Kombinationen - Stabil vs Instabil | " + name + "\n"
                    f"Gross: F={[int(f) for f in reversed(F_gross)]} | "
                    f"Klein: F={[int(f) for f in reversed(F_klein)]}")
        fig_all.suptitle(titel_all, color="#cccccc", fontsize=11, y=1.01)
        plt.tight_layout()
        out_all = out_dir / f"{name}_tabellen_kombinationen.png"
        fig_all.savefig(out_all, dpi=120, bbox_inches="tight", facecolor="#0d0d0d")
        plt.close(fig_all)
        print(f"[viz_matrix] Alle Tabellen gespeichert: {out_all}")


    # ── Aktuellen Zustand ausgeben ───────────────────────────
    x_curr = tuple(int(dom_klein[F][-1]) for F in F_klein)
    y_curr = tuple(int(dom_gross[F][-1]) for F in F_gross)
    print(f"[viz_matrix] Aktueller Zustand:")
    print(f"  Groß  (Y): {_state_to_label(y_curr)} = {y_curr}")
    print(f"  Klein (X): {_state_to_label(x_curr)} = {x_curr}")
    xi = state_to_idx[x_curr]
    yi = state_to_idx[y_curr]
    print(f"  Theo:  {theo_matrix[yi,xi]:+.1%}")
    if not np.isnan(hist_norm[yi,xi]):
        print(f"  Hist:  {hist_norm[yi,xi]:+.1%}")
        print(f"  Delta: {diff_matrix[yi,xi]:+.2f}")
        print(f"  Anzahl Ticks in diesem Zustand: {count_matrix[yi,xi]:,}")
