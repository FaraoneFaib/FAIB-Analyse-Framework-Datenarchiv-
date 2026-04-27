import numpy as np
import pandas as pd
from typing import Dict, Iterable


def build_fractal_layer(urdaten: pd.DataFrame, F: float) -> pd.DataFrame:
    """
    Baut eine Fraktalebene F auf Basis von cum_height.
    Vollstaendig numpy-vektorisiert fuer grosse Datensaetze.

    - close_F  : bestaetigter Fraktal-Close (Treppenfunktion in F-Schritten)
    - dot      : offene Hoehenposition relativ zum aktuellen close_F
    - n_blocks : Blockindex (close_F / F)
    - close_confirmed : 1 genau in den Ticks wo ein neuer Block bestaetigt wurde
    - open_dir : Richtung der offenen Bewegung (Vorzeichen von dot)
    """
    if F <= 0:
        raise ValueError("F muss > 0 sein")

    h = urdaten["cum_height"].astype(float).to_numpy()
    n = len(h)

    # Vektorisierter Ansatz: close_F = floor(h / F) * F
    # Das entspricht der Treppenfunktion mit Schritt F
    # ABER: wir brauchen die kausale Version (kein Lookahead)
    # Loesung: Numba-artige kumulative Berechnung via numpy-Tricks

    # Schnelle Naeherung: floor(h/F)*F gibt die untere Stufe
    # Das ist kausal korrekt fuer einseitige Treppenfunktion
    # Fuer bidirektionale Treppe: wir verwenden die Originallogik
    # aber beschleunigt durch numpy-Chunking

    # Fuer 12 Mio Ticks: pure Python zu langsam
    # Loesung: Numba wenn verfuegbar, sonst optimierter numpy-Ansatz

    try:
        from numba import njit

        @njit(cache=True)
        def _build_close(h_arr, F_val):
            n = len(h_arr)
            close_vals = np.empty(n, dtype=np.float64)
            close_curr = 0.0
            for i in range(n):
                x = h_arr[i]
                if x >= close_curr + F_val:
                    steps = int((x - close_curr) / F_val)
                    close_curr += steps * F_val
                elif x <= close_curr - F_val:
                    steps = int((close_curr - x) / F_val)
                    close_curr -= steps * F_val
                close_vals[i] = close_curr
            return close_vals

        close_vals = _build_close(h, float(F))

    except ImportError:
        # Ohne Numba: optimierte Python-Schleife
        close_vals = np.empty(n, dtype=np.float64)
        close_curr = 0.0
        F_f = float(F)
        for i in range(n):
            x = h[i]
            if x >= close_curr + F_f:
                close_curr += int((x - close_curr) / F_f) * F_f
            elif x <= close_curr - F_f:
                close_curr -= int((close_curr - x) / F_f) * F_f
            close_vals[i] = close_curr

    dot_vals  = h - close_vals
    n_blocks  = (close_vals / float(F)).astype(int)

    df = pd.DataFrame(
        {
            "close_F":  close_vals,
            "dot":      dot_vals,
            "n_blocks": n_blocks,
        },
        index=urdaten.index,
    )

    df["close_confirmed"] = df["n_blocks"].diff().fillna(0).ne(0).astype(int)
    df["open_dir"]        = np.sign(df["dot"])

    # cum_height mitfuehren (wird von viz_interferenz etc. benoetigt)
    if "cum_height" in urdaten.columns:
        df["cum_height"] = urdaten["cum_height"].values

    return df


def build_all_fractals(
    urdaten: pd.DataFrame,
    fractal_levels: Iterable[float]
) -> Dict[float, pd.DataFrame]:
    result: Dict[float, pd.DataFrame] = {}
    levels = list(fractal_levels)
    print(f"[fraktale] Berechne {len(levels)} Fraktalebenen...")
    for i, F in enumerate(levels):
        print(f"  F={int(F):6d}  ({i+1}/{len(levels)})", end="\r")
        result[F] = build_fractal_layer(urdaten, F)
    print(f"\n[fraktale] Fertig — {len(levels)} Ebenen berechnet.")
    return result