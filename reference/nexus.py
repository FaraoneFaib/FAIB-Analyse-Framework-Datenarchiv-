import pandas as pd
from typing import Dict

def build_nexus_matrix(dom_by_F: Dict[int, pd.Series]) -> pd.DataFrame:
    """
    dom_by_F: {Fraktalhöhe F: Dominanz-Serien (0/1)}
    NEXUS-Matrix: Spalten mit Namen "F{F}"
    """
    df = pd.DataFrame(index=next(iter(dom_by_F.values())).index)
    for F, dom in sorted(dom_by_F.items(), key=lambda x: x[0]):
        df[f"F{F}"] = dom
    return df

def fusion(nexus_df: pd.DataFrame) -> pd.Series:
    """
    Fusion: Summe aller Dominanzrichtungen.
    Annahme: 1 = Long, 0 = Short -> Übersetzung in +1 / -1.
    """
    # +1 für Long, -1 für Short
    signed = nexus_df.replace({1: 1, 0: -1})
    return signed.sum(axis=1)

def verlag(fusion_series: pd.Series) -> pd.Series:
    """Verlagerung: zeitliche Ableitung der Fusion."""
    return fusion_series.diff().fillna(0.0)