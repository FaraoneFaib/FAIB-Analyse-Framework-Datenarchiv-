from pathlib import Path
import pandas as pd

from config import (
    SYSTEMS,
    FRACTAL_LEVELS,
    MACD_FAST,
    MACD_SLOW,
    OUTPUT_DIR,
)
import loaders
from urdaten import build_urdaten
from fraktale import build_all_fractals
from macd_dom import macd, macd_on_change, dominance_binary
from nexus import build_nexus_matrix, fusion, verlag
import main_storage
from viz import (
    plot_urdaten_debug,
    plot_cumheight_with_fractals,
    plot_fractal_closes_grid,
    plot_nexus_debug,
    plot_all_single_fractals,
    plot_nexus,
)

# ============================================================
# MODUS
# "full"       = alles neu berechnen + speichern + Basisplots
# "test_cache" = nur laden + optionale Module/Plots
# "cache_plot" = laden + Basisplots + optionale Module
# ============================================================
MODE = "full"

# ============================================================
# BASIS-PLOTS SCHALTER
# ============================================================
ENABLE_PLOT_01_URDATEN     = False
ENABLE_PLOT_02_CUMHEIGHT   = False
ENABLE_PLOT_03_GRID        = False
ENABLE_PLOT_04_NEXUS_DEBUG = False
ENABLE_PLOT_05_SINGLE      = False

# ============================================================
# OPTIONALE MODULE
# ============================================================
ENABLE_RS              = False
ENABLE_DFA             = False
ENABLE_MFDFA           = False # Code vorerst ohen Anwendung
ENABLE_HURST           = False # leer
ENABLE_SIGNALS_EXTRA   = False # leer
ENABLE_VIZ_COMPARE     = False
ENABLE_VIZ_PANELS      = False
ENABLE_FEATURES        = False
ENABLE_VIZ_INTERACTIVE = False # ersetzt durch Plotly
ENABLE_VIZ_PLOTLY      = True
ENABLE_VIZ_VERLAGERUNG = False
ENABLE_VIZ_MATRIX      = False
ENABLE_VIZ_INTERFERENZ = False
ENABLE_VIZ_HURST       = False

# ============================================================
# OPTIONALE IMPORTE
# ============================================================
try:
    import rs
except ImportError:
    rs = None

try:
    import dfa
except ImportError:
    dfa = None

try:
    import mfdfa
except ImportError:
    mfdfa = None

try:
    import hurst
except ImportError:
    hurst = None

try:
    import signals_extra
except ImportError:
    signals_extra = None

try:
    import viz_compare
except ImportError:
    viz_compare = None

try:
    import viz_panels
except ImportError:
    viz_panels = None

try:
    import features
except ImportError:
    features = None

try:
    import viz_interactive
except ImportError:
    viz_interactive = None

try:
    import viz_plotly
except ImportError:
    viz_plotly = None

try:
    import viz_verlagerung
except ImportError:
    viz_verlagerung = None

try:
    import viz_matrix
except ImportError:
    viz_matrix = None

try:
    import viz_interferenz
except ImportError:
    viz_interferenz = None

try:
    import viz_hurst
except ImportError:
    viz_hurst = None

MACD_SIGNAL     = 5
DOT_SMOOTH_SPAN = 3


def run_optional_features(name, ur, fractals, nexus_df,
                           fusion_series, verlag_series, out_dir):
    """
    Berechnet alle optionalen Features und gibt Ergebnisse zurueck.
    rs_res, dfa_res, mfdfa_res werden als dict zurueckgegeben
    damit run_optional_views sie verwenden kann.
    """
    rs_res    = {}
    dfa_res   = {}
    mfdfa_res = {}

    if ENABLE_RS and rs is not None and hasattr(rs, "run_rs"):
        rs_res = rs.run_rs(name, ur, fractals, out_dir)

    if ENABLE_DFA and dfa is not None and hasattr(dfa, "run_dfa"):
        dfa_res = dfa.run_dfa(name, ur, fractals, out_dir)

    if ENABLE_MFDFA and mfdfa is not None and hasattr(mfdfa, "run_mfdfa"):
        mfdfa_res = mfdfa.run_mfdfa(name, ur, fractals, out_dir)

    if ENABLE_HURST and hurst is not None and hasattr(hurst, "run_hurst"):
        hurst.run_hurst(name, ur, fractals, out_dir)

    if ENABLE_SIGNALS_EXTRA and signals_extra is not None and hasattr(signals_extra, "run_signals_extra"):
        signals_extra.run_signals_extra(name, ur, fractals, out_dir)

    if ENABLE_FEATURES and features is not None and hasattr(features, "run_features"):
        features.run_features(name, ur, fractals, nexus_df,
                              fusion_series, verlag_series, out_dir)

    # Ergebnisse zurueckgeben fuer viz_hurst
    return rs_res, dfa_res, mfdfa_res


def run_basis_plots(name, ur, fractals, nexus_df,
                    fusion_series, verlag_series, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)

    if ENABLE_PLOT_01_URDATEN:
        print(f"  → _01_urdaten_debug")
        plot_urdaten_debug(name, ur, out_dir)

    if ENABLE_PLOT_02_CUMHEIGHT:
        print(f"  → _02_cumheight_fractals")
        plot_cumheight_with_fractals(name, ur, fractals, out_dir)

    if ENABLE_PLOT_03_GRID:
        print(f"  → _03_fractal_closes_grid")
        plot_fractal_closes_grid(name, fractals, out_dir)

    if ENABLE_PLOT_04_NEXUS_DEBUG:
        print(f"  → _04_nexus_debug")
        plot_nexus_debug(name, nexus_df, fusion_series, verlag_series,
                         out_dir, fractals=fractals, urdaten=ur,
                         verlag_full=verlag_series)

    if ENABLE_PLOT_05_SINGLE:
        print(f"  → _fraktal_FX_chart (alle Fraktale)")
        plot_all_single_fractals(name, ur, fractals, out_dir)


def run_optional_views(name, ur, fractals, nexus_df,
                       fusion_series, verlag_series, out_dir,
                       rs_res=None, dfa_res=None, mfdfa_res=None):
    """
    rs_res, dfa_res, mfdfa_res kommen von run_optional_features.
    """
    rs_res    = rs_res    or {}
    dfa_res   = dfa_res   or {}
    mfdfa_res = mfdfa_res or {}

    if ENABLE_VIZ_COMPARE and viz_compare is not None and hasattr(viz_compare, "plot_compare"):
        viz_compare.plot_compare(name, ur, fractals, nexus_df,
                                 fusion_series, verlag_series, out_dir)

    if ENABLE_VIZ_PANELS and viz_panels is not None and hasattr(viz_panels, "plot_panels"):
        viz_panels.plot_panels(name, ur, fractals, nexus_df,
                               fusion_series, verlag_series, out_dir)

    if ENABLE_VIZ_INTERACTIVE and viz_interactive is not None:
        viz_interactive.plot_interactive(name, ur, fractals, nexus_df, out_dir)

    if ENABLE_VIZ_PLOTLY and viz_plotly is not None:
        viz_plotly.plot_plotly(name, ur, fractals, nexus_df, out_dir)

    if ENABLE_VIZ_VERLAGERUNG and viz_verlagerung is not None:
        viz_verlagerung.plot_verlagerung(name, ur, fractals, nexus_df,
                                         fusion_series, verlag_series, out_dir)

    if ENABLE_VIZ_MATRIX and viz_matrix is not None:
        viz_matrix.plot_matrix(name, ur, fractals, nexus_df, out_dir)

    if ENABLE_VIZ_INTERFERENZ and viz_interferenz is not None:
        viz_interferenz.plot_interferenz(name, ur, fractals, out_dir)

    if ENABLE_VIZ_HURST and viz_hurst is not None:
        viz_hurst.plot_hurst(name, ur, fractals,
                             rs_res, dfa_res, mfdfa_res, out_dir)


def save_base_outputs(ur, fractals, nexus_df, fusion_series,
                      verlag_series, out_dir):
    import config as _cfg
    main_storage.save_base_outputs(
        ur, fractals, nexus_df, fusion_series, verlag_series,
        out_dir, config=_cfg)


def load_base_outputs(out_dir, fractal_levels):
    import config as _cfg
    return main_storage.load_base_outputs(
        out_dir, fractal_levels, config=_cfg)


def build_base_data(series):
    ur       = build_urdaten(series, mode="raw", scale_factor=1.0, center=False)
    fractals = build_all_fractals(ur, FRACTAL_LEVELS)
    dom_by_F = {}

    for F, dfF in fractals.items():
        macd_close    = macd_on_change(dfF["close_F"], fast=MACD_FAST,
                                       slow=MACD_SLOW, signal_span=MACD_SIGNAL)
        fractal_state = dfF["n_blocks"].astype(float) + (dfF["dot"] / float(F))
        macd_state    = macd(fractal_state, fast=MACD_FAST,
                             slow=MACD_SLOW, signal_span=MACD_SIGNAL)
        dot_smooth    = dfF["dot"].ewm(span=DOT_SMOOTH_SPAN, adjust=False).mean()

        df_out = dfF.copy()
        df_out["dot_smooth"]    = dot_smooth
        df_out["fractal_state"] = fractal_state
        df_out = pd.concat([df_out,
                            macd_close.add_prefix("c_"),
                            macd_state.add_prefix("s_")], axis=1)

        dom = dominance_binary(macd_close)
        df_out["dominance"] = dom
        fractals[F]  = df_out
        dom_by_F[F]  = dom

    nexus_df      = build_nexus_matrix(dom_by_F)
    fusion_series = fusion(nexus_df)
    verlag_series = verlag(fusion_series)

    return ur, fractals, nexus_df, fusion_series, verlag_series


def run_system(system_conf: dict):
    name        = system_conf["name"]
    file        = system_conf["file"]
    loader_name = system_conf["loader"]

    print(f"\n=== FAIB-NEXUS V2: {name} | MODE={MODE} ===")

    out_dir = OUTPUT_DIR / f"faib_{name}_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    if MODE == "full":
        if not file.exists():
            print(f"Datei nicht gefunden: {file}")
            return
        loader_func = getattr(loaders, loader_name)
        series = loader_func(file)
        ur, fractals, nexus_df, fusion_series, verlag_series = build_base_data(series)
        save_base_outputs(ur, fractals, nexus_df, fusion_series,
                          verlag_series, out_dir)
        run_basis_plots(name, ur, fractals, nexus_df,
                        fusion_series, verlag_series, out_dir)

    elif MODE == "test_cache":
        ur, fractals, nexus_df, fusion_series, verlag_series = \
            load_base_outputs(out_dir, FRACTAL_LEVELS)

    elif MODE == "cache_plot":
        ur, fractals, nexus_df, fusion_series, verlag_series = \
            load_base_outputs(out_dir, FRACTAL_LEVELS)
        run_basis_plots(name, ur, fractals, nexus_df,
                        fusion_series, verlag_series, out_dir)

    else:
        raise ValueError(f"Unbekannter MODE: {MODE}")

    # Features berechnen — Ergebnisse weitergeben
    rs_res, dfa_res, mfdfa_res = run_optional_features(
        name, ur, fractals, nexus_df,
        fusion_series, verlag_series, out_dir)

    # Views mit Feature-Ergebnissen
    run_optional_views(
        name, ur, fractals, nexus_df,
        fusion_series, verlag_series, out_dir,
        rs_res=rs_res, dfa_res=dfa_res, mfdfa_res=mfdfa_res)

    print(f"Fertig: {name} → {out_dir}")


def main():
    for sys_conf in SYSTEMS:
        run_system(sys_conf)


if __name__ == "__main__":
    main()