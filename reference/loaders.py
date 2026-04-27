import re
import pandas as pd
from pathlib import Path


# ============================================================
# SOLAR LOADER
# ============================================================

def load_solar_daily(path: Path) -> pd.Series:
    """
    Laedt taeglich Sonnenflecken-Daten (SIDC Format).
    Format: year month day decimal_year sunspot_number std n_obs definitive
    -1 = keine Messung → wird gefiltert
    """
    cols = ["year", "month", "day", "decimal_year",
            "sunspot_number", "std", "n_obs", "definitive"]

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=cols,
        engine="python",
        on_bad_lines="skip"
    )

    for c in ["year", "month", "day", "sunspot_number"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["year", "month", "day", "sunspot_number"]).copy()

    # -1 = keine Messung entfernen
    df = df[df["sunspot_number"] >= 0].copy()

    df[["year", "month", "day"]] = df[["year", "month", "day"]].astype(int)
    df["date"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=df["day"]),
        errors="coerce"
    )
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").set_index("date")

    print(f"[Solar] Geladen: {len(df):,} Tage | "
          f"{df.index[0].date()} → {df.index[-1].date()}")
    print(f"[Solar] Min={df['sunspot_number'].min():.0f} "
          f"Max={df['sunspot_number'].max():.0f} "
          f"Mean={df['sunspot_number'].mean():.1f}")

    return df["sunspot_number"].astype(float)


# ============================================================
# NASDAQ SINGLE FILE LOADER
# ============================================================

def _parse_nasdaq_file(path: Path) -> pd.DataFrame:
    """
    Lädt eine einzelne NASDAQ L2-Tick-Datei.
    Format: ts;level;type;price;volume;action;depth
    Gibt einen sauberen DataFrame zurück oder None bei Fehler.
    """
    try:
        df = pd.read_csv(
            path,
            header=None,
            sep=";",
            engine="c",
            dtype=str,
            on_bad_lines="skip",
        )
    except Exception as e:
        print(f"  [WARN] Fehler beim Lesen von {path.name}: {e}")
        return None

    if df.shape[1] < 4:
        print(f"  [WARN] Zu wenige Spalten in {path.name} ({df.shape[1]}) — übersprungen")
        return None

    n_cols = min(df.shape[1], 7)
    df = df.iloc[:, :n_cols].copy()

    base_cols = ["ts", "level", "type", "price", "volume", "action", "depth"]
    df.columns = base_cols[:n_cols]

    for col in ["level", "type", "price"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    needed = ["ts", "price"]
    df = df.dropna(subset=[c for c in needed if c in df.columns]).copy()

    # Nur Last Trade (Type=2) — Level-Filter weggelassen
    # L2-Dateien haben Level=2 durchgehend, nicht relevant für Preis
    if "type" in df.columns:
        df = df[df["type"] == 2].copy()

    if df.empty:
        print(f"  [WARN] Keine gültigen Ticks in {path.name} — übersprungen")
        return None

    df["ts"] = df["ts"].astype(str)
    # MarketTick hat Mikrosekunden (20 Stellen) — wir schneiden auf Sekunde
    # Fuer FAIB fraktal-basierte Analyse ist Sekundengenauigkeit ausreichend
    df["ts"] = pd.to_datetime(
        df["ts"].str.slice(0, 14),
        format="%Y%m%d%H%M%S",
        errors="coerce",
    )

    df = df.dropna(subset=["ts"]).copy()

    if df.empty:
        print(f"  [WARN] Keine gültigen Zeitstempel in {path.name} — übersprungen")
        return None

    df = df.sort_values("ts").set_index("ts")
    return df[["price"]]


def _extract_date_from_filename(filename: str):
    """
    Extrahiert Datum aus Dateinamen wie:
    20200614.csv → 20200614
    20200801.csv → 20200801
    nasdaq_20200614.csv → 20200614
    Gibt ein sortierbares Datum-Objekt zurück oder None.
    """
    # Suche nach 8-stelliger Zahl (YYYYMMDD)
    match = re.search(r'(\d{8})', filename)
    if match:
        date_str = match.group(1)
        try:
            return pd.to_datetime(date_str, format="%Y%m%d")
        except Exception:
            return None
    return None


def load_nasdaq_series(path: Path) -> pd.Series:
    """
    Einzelne NASDAQ Datei laden.
    """
    df = _parse_nasdaq_file(path)
    if df is None or df.empty:
        raise ValueError(f"Keine gültigen Daten in {path}")
    return df["price"].astype(float)


def load_nasdaq_multiday(folder: Path) -> pd.Series:
    """
    Lädt ALLE NASDAQ CSV-Dateien aus einem Ordner.

    - Erkennt Dateien automatisch nach Datum im Dateinamen (YYYYMMDD)
    - Sortiert chronologisch
    - Überspringt fehlerhafte Dateien mit Warnung
    - Entfernt Duplikate
    - Gibt eine vollständige Preisserie zurück

    Dateinamen-Beispiele die erkannt werden:
        20200614.csv
        20200801.csv
        nasdaq_20200614.csv
        MNQ_20200614_ticks.csv
    """
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"Ordner nicht gefunden: {folder}")

    # Alle CSV-Dateien finden
    # Deduplizieren: glob kann gleiche Dateien doppelt finden
    seen = set()
    all_files = []
    for f in list(folder.glob("*.csv")) + list(folder.glob("*.CSV")):
        if f.resolve() not in seen:
            seen.add(f.resolve())
            all_files.append(f)

    if not all_files:
        raise ValueError(f"Keine CSV-Dateien in {folder}")

    # Datum extrahieren und sortieren
    dated_files = []
    skipped = []

    for f in all_files:
        date = _extract_date_from_filename(f.name)
        if date is not None:
            dated_files.append((date, f))
        else:
            skipped.append(f.name)

    if skipped:
        print(f"[INFO] {len(skipped)} Dateien ohne erkennbares Datum übersprungen:")
        for s in skipped:
            print(f"  → {s}")

    if not dated_files:
        raise ValueError(f"Keine Dateien mit erkennbarem Datum (YYYYMMDD) in {folder}")

    # Chronologisch sortieren
    dated_files.sort(key=lambda x: x[0])

    print(f"\n[NASDAQ Multiday] Lade {len(dated_files)} Dateien aus {folder.name}/")
    print(f"  Von: {dated_files[0][0].strftime('%Y-%m-%d')}")
    print(f"  Bis: {dated_files[-1][0].strftime('%Y-%m-%d')}")
    print()

    # Alle Dateien laden
    all_dfs = []
    loaded = 0
    failed = 0

    for date, filepath in dated_files:
        print(f"  Lade {filepath.name} ({date.strftime('%Y-%m-%d')})...", end=" ")
        df = _parse_nasdaq_file(filepath)

        if df is not None and not df.empty:
            all_dfs.append(df)
            print(f"✓ {len(df):,} Ticks")
            loaded += 1
        else:
            print("✗ übersprungen")
            failed += 1

    print(f"\n[NASDAQ Multiday] {loaded} Dateien geladen, {failed} übersprungen")

    if not all_dfs:
        raise ValueError("Keine gültigen Daten in allen Dateien!")

    # Zusammenführen
    combined = pd.concat(all_dfs)
    combined = combined.sort_index()

    # Duplikate entfernen
    dupes = combined.index.duplicated().sum()
    if dupes > 0:
        print(f"[INFO] {dupes:,} doppelte Zeitstempel entfernt")
        combined = combined[~combined.index.duplicated(keep="first")]

    total_ticks = len(combined)
    print(f"[NASDAQ Multiday] Gesamt: {total_ticks:,} Ticks")
    print(f"  Zeitraum: {combined.index[0]} → {combined.index[-1]}")

    return combined["price"].astype(float)

# ============================================================
# ERA5 WIND LOADER (NetCDF4 / xarray)
# ============================================================

def load_era5_wind(path: Path) -> pd.Series:
    """
    Laedt ERA5 Winddaten aus einer NetCDF4 Datei.
    Berechnet Windgeschwindigkeit aus u10 und v10 Komponenten.

    Konfiguration in config.py unter ERA5_CONFIG:
    → lat/lon: Gitterpunkt (naechster Punkt wird gewaehlt)
    → variables: ["u10", "v10"] fuer Wind
    → combine: "wind_speed" = sqrt(u² + v²)
    """
    try:
        import xarray as xr
    except ImportError:
        raise ImportError(
            "xarray nicht installiert!\n"
            "Bitte installieren: pip install xarray netCDF4"
        )

    import numpy as np

    # Konfiguration aus config laden
    try:
        from config import ERA5_CONFIG
        lat = ERA5_CONFIG.get("lat", 35.5)
        lon = ERA5_CONFIG.get("lon", -97.5)
        variables = ERA5_CONFIG.get("variables", ["u10", "v10"])
        combine = ERA5_CONFIG.get("combine", "wind_speed")
    except ImportError:
        lat, lon = 35.5, -97.5
        variables = ["u10", "v10"]
        combine = "wind_speed"

    print(f"[ERA5] Lade {path.name}")
    print(f"       Punkt: lat={lat}, lon={lon}")
    print(f"       Variablen: {variables}")

    ds = xr.open_dataset(path)

    # Zeitachse
    time_idx = ds.valid_time.values

    # Variablen extrahieren am naechsten Gitterpunkt
    if combine == "wind_speed" and len(variables) == 2:
        # sqrt(u² + v²)
        u = ds[variables[0]].sel(
            latitude=lat, longitude=lon, method="nearest"
        ).values.astype(float)
        v = ds[variables[1]].sel(
            latitude=lat, longitude=lon, method="nearest"
        ).values.astype(float)
        values = np.sqrt(u**2 + v**2)
        print(f"[ERA5] Windgeschwindigkeit: min={values.min():.2f} "
              f"max={values.max():.2f} mean={values.mean():.2f} m/s")
    else:
        # Einzelne Variable
        values = ds[variables[0]].sel(
            latitude=lat, longitude=lon, method="nearest"
        ).values.astype(float)
        print(f"[ERA5] {variables[0]}: min={values.min():.2f} "
              f"max={values.max():.2f} mean={values.mean():.2f}")

    ds.close()

    series = pd.Series(values, index=pd.DatetimeIndex(time_idx), name=combine)
    series = series.sort_index().dropna()

    print(f"[ERA5] Gesamt: {len(series):,} Datenpunkte")
    print(f"       Zeitraum: {series.index[0]} → {series.index[-1]}")

    return series

# ============================================================
# Stellare Lichtkurve Datenanalyse KIC 6922244 (Kepler-8)
# Start: BKJD 169.53 (~Juni 2009)
# Ende: BKJD 1273.07 (~Oktober 2012)
# Aufnahmen im ~58,85 Sekunden-Takt (1-Minuten-Intervall).
# ============================================================



def load_kic_time_flux(path: Path) -> pd.Series:
    """
    Lädt KIC 6922244 Zeit-Fluss-Datei (2-Spalten-Text).
    Erwartet:
        Spalte 1: Zeit_BKJD (Tage seit 01.01.2009)
        Spalte 2: Lichtfluss_PDCSAP (e-/s)
    Gibt eine Serie mit Zeitindex und Flusswerten zurück.
    """
    path = Path(path)

    df = pd.read_csv(
        path,
        sep=r"\s+|,|;",           # flexibel: Leerzeichen, Tab, Komma, Semikolon
        engine="python",
        header=None,
        names=["Zeit_BKJD", "Lichtfluss_PDCSAP"],
        comment="#",
    )

    # NaNs entfernen
    df = df.dropna(subset=["Zeit_BKJD", "Lichtfluss_PDCSAP"]).copy()

    # Numerisch casten
    df["Zeit_BKJD"] = pd.to_numeric(df["Zeit_BKJD"], errors="coerce")
    df["Lichtfluss_PDCSAP"] = pd.to_numeric(df["Lichtfluss_PDCSAP"], errors="coerce")
    df = df.dropna(subset=["Zeit_BKJD", "Lichtfluss_PDCSAP"]).copy()

    # Zeit als "Pseudo-Datetime" – BKJD ist in Tagen, wir mappen das auf eine
    # künstliche Startzeit, damit Matplotlib/FAIB sauber plotten kann.
    # 2009-01-01 + BKJD Tage:
    start = pd.Timestamp("2009-01-01")
    df["time"] = start + pd.to_timedelta(df["Zeit_BKJD"], unit="D")

    df = df.sort_values("time").set_index("time")

    print(
        f"[KIC 6922244] Geladen: {len(df):,} Ticks | "
        f"{df.index[0]} → {df.index[-1]} | "
        f"Flux-Min={df['Lichtfluss_PDCSAP'].min():.1f} "
        f"Max={df['Lichtfluss_PDCSAP'].max():.1f}"
    )

    # Rückgabe: nur Fluss als Serie (für build_urdaten)
    return df["Lichtfluss_PDCSAP"].astype(float)