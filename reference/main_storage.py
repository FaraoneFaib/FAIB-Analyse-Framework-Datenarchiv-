"""
FAIB NEXUS — Storage-Modul
==========================
Flexibel zwischen CSV und Parquet wechselbar.
Schalter in config.py: STORAGE_FORMAT = "parquet" | "csv"
"""

import pandas as pd
from pathlib import Path


def _fmt(config_module):
    """Liest STORAGE_FORMAT aus config, default = 'csv'."""
    return getattr(config_module, "STORAGE_FORMAT", "csv").lower()


def save_base_outputs(ur, fractals, nexus_df, fusion_series,
                      verlag_series, out_dir, config=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = _fmt(config) if config else "csv"

    print(f"[storage] Speichere als .{fmt} ...")

    if fmt == "parquet":
        try:
            import pyarrow
        except ImportError:
            print("[storage] WARNUNG: pyarrow nicht installiert!")
            print("[storage] Fallback auf CSV. Installiere: pip install pyarrow")
            fmt = "csv"

    if fmt == "parquet":
        ur.to_parquet(out_dir / "urdaten.parquet")

        for F, dfF in fractals.items():
            dfF.to_parquet(out_dir / f"fraktal_F{int(F)}.parquet")

            # Events weiterhin als CSV (klein, gut lesbar)
            if "close_confirmed" in dfF.columns:
                events = dfF[dfF["close_confirmed"] == 1].copy()
                cols = ["n_blocks","close_F","dot","fractal_state",
                        "c_macd","c_signal","c_hist",
                        "s_macd","s_signal","s_hist"]
                cols = [c for c in cols if c in events.columns]
                if cols:
                    events[cols].to_csv(
                        out_dir / f"fraktal_F{int(F)}_events.csv",
                        index=True)

        nexus_df.to_parquet(out_dir / "nexus_matrix.parquet")

        fv = pd.DataFrame({
            "fusion":      fusion_series,
            "verlagerung": verlag_series
        })
        fv.to_parquet(out_dir / "fusion_verlagerung.parquet")

    else:
        # CSV — Originalverhalten
        ur.to_csv(out_dir / "urdaten.csv", index=True)

        for F, dfF in fractals.items():
            dfF.to_csv(out_dir / f"fraktal_F{int(F)}.csv", index=True)

            if "close_confirmed" in dfF.columns:
                events = dfF[dfF["close_confirmed"] == 1].copy()
                cols = ["n_blocks","close_F","dot","fractal_state",
                        "c_macd","c_signal","c_hist",
                        "s_macd","s_signal","s_hist"]
                cols = [c for c in cols if c in events.columns]
                if cols:
                    events[cols].to_csv(
                        out_dir / f"fraktal_F{int(F)}_events.csv",
                        index=True)

        nexus_df.to_csv(out_dir / "nexus_matrix.csv", index=True)

        pd.DataFrame({
            "fusion":      fusion_series,
            "verlagerung": verlag_series
        }).to_csv(out_dir / "fusion_verlagerung.csv", index=True)

    # Meta immer als CSV
    import pandas as pd2
    meta_df = pd2.DataFrame({
        "parameter": ["storage_format", "macd_fast", "macd_slow",
                       "macd_signal", "n_fractals"],
        "value":     [fmt, 6, 26, 5,
                       len(fractals)],
    })
    meta_df.to_csv(out_dir / "run_meta.csv", index=False)
    print(f"[storage] Fertig — Format: {fmt}")


def load_base_outputs(out_dir, fractal_levels, config=None):
    fmt = _fmt(config) if config else "csv"

    # Auto-Erkennung: wenn Parquet-Dateien da sind, nimm die
    ur_parquet = out_dir / "urdaten.parquet"
    ur_csv     = out_dir / "urdaten.csv"

    if ur_parquet.exists():
        fmt_actual = "parquet"
    elif ur_csv.exists():
        fmt_actual = "csv"
    else:
        raise FileNotFoundError(
            f"Keine urdaten.parquet oder urdaten.csv in {out_dir}")

    if fmt_actual != fmt:
        print(f"[storage] Hinweis: config sagt '{fmt}' "
              f"aber gefunden: '{fmt_actual}' — nehme '{fmt_actual}'")
    fmt = fmt_actual

    print(f"[storage] Lade als .{fmt} ...")

    if fmt == "parquet":
        ur       = pd.read_parquet(ur_parquet)
        nexus_df = pd.read_parquet(out_dir / "nexus_matrix.parquet")
        fv_df    = pd.read_parquet(out_dir / "fusion_verlagerung.parquet")

        fractals = {}
        for F in fractal_levels:
            p = out_dir / f"fraktal_F{int(F)}.parquet"
            if p.exists():
                fractals[F] = pd.read_parquet(p)

    else:
        ur       = pd.read_csv(out_dir / "urdaten.csv",     index_col=0)
        nexus_df = pd.read_csv(out_dir / "nexus_matrix.csv", index_col=0)
        fv_df    = pd.read_csv(out_dir / "fusion_verlagerung.csv",
                               index_col=0)

        fractals = {}
        for F in fractal_levels:
            p = out_dir / f"fraktal_F{int(F)}.csv"
            if p.exists():
                fractals[F] = pd.read_csv(p, index_col=0)

    fusion_series = (fv_df["fusion"]      if "fusion"      in fv_df.columns
                     else pd.Series(dtype=float))
    verlag_series = (fv_df["verlagerung"] if "verlagerung" in fv_df.columns
                     else pd.Series(dtype=float))

    print(f"[storage] Geladen: {len(fractals)} Fraktale")
    return ur, fractals, nexus_df, fusion_series, verlag_series