import matplotlib.pyplot as plt
import numpy as np


def _safe_sample(df, max_points=1800):
    if len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()


def plot_compare(name, ur, fractals, nexus_df, fusion_series, verlag_series, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for F in sorted(fractals.keys()):
        dfF = fractals[F].copy().reset_index(drop=True)
        plot_df = _safe_sample(dfF, max_points=1800)

        required_cols = [
            "c_macd", "c_signal", "c_hist",
            "s_macd", "s_signal", "s_hist",
        ]
        if not all(col in plot_df.columns for col in required_cols):
            continue

        x = np.arange(len(plot_df))

        fig, axes = plt.subplots(
            2, 1,
            figsize=(15, 8.5),
            sharex=True,
            gridspec_kw={"height_ratios": [1, 1]}
        )

        ax1, ax2 = axes

        # ---------------------------------------------------
        # Oberes Panel: c_* = Struktur-MACD auf close_F
        # ---------------------------------------------------
        c_hist = plot_df["c_hist"].fillna(0.0).to_numpy()
        c_macd = plot_df["c_macd"].fillna(0.0).to_numpy()
        c_signal = plot_df["c_signal"].fillna(0.0).to_numpy()

        c_colors = ["darkgreen" if h >= 0 else "darkred" for h in c_hist]

        ax1.bar(x, c_hist, color=c_colors, alpha=0.35, width=0.8, label="c_hist")
        ax1.plot(x, c_macd, color="tab:blue", lw=1.1, label="c_macd")
        ax1.plot(x, c_signal, color="black", lw=1.0, alpha=0.9, label="c_signal")
        ax1.axhline(0, color="black", lw=0.8, alpha=0.6)

        ax1.set_title(f"Struktur-MACD auf close_F (on_change), F={F}")
        ax1.set_ylabel("c_*")
        ax1.grid(True, alpha=0.2)
        ax1.legend(loc="best", fontsize=8)

        # ---------------------------------------------------
        # Unteres Panel: s_* = Zustands-MACD auf fractal_state
        # ---------------------------------------------------
        s_hist = plot_df["s_hist"].fillna(0.0).to_numpy()
        s_macd = plot_df["s_macd"].fillna(0.0).to_numpy()
        s_signal = plot_df["s_signal"].fillna(0.0).to_numpy()

        s_colors = ["darkgreen" if h >= 0 else "darkred" for h in s_hist]

        ax2.bar(x, s_hist, color=s_colors, alpha=0.35, width=0.8, label="s_hist")
        ax2.plot(x, s_macd, color="tab:blue", lw=1.1, label="s_macd")
        ax2.plot(x, s_signal, color="black", lw=1.0, alpha=0.9, label="s_signal")
        ax2.axhline(0, color="black", lw=0.8, alpha=0.6)

        ax2.set_title(f"Zustands-MACD auf fractal_state, F={F}")
        ax2.set_ylabel("s_*")
        ax2.set_xlabel("Zeit / Index")
        ax2.grid(True, alpha=0.2)
        ax2.legend(loc="best", fontsize=8)

        fig.suptitle(f"{name} – Vergleich c_* vs. s_* bei F={F}", y=0.995, fontsize=13)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}_compare_c_vs_s_F{int(F)}.png", dpi=160)
        plt.close(fig)