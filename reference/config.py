from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

SOLAR_FILE    = BASE_DIR / "data" / "SN_d_tot_V2.0.txt"
NASDAQ_FILE   = BASE_DIR / "data" / "20200918.csv"
NASDAQ_FOLDER = BASE_DIR / "data" / "nasdaq_multiday"
ERA5_FILE     = BASE_DIR / "data" / "data_stream-oper_stepType-instant.nc"
KIC6922244_FILE  = BASE_DIR / "data" / "kic6922244_zeit_und_fluss.txt"

# ============================================================
# ERA5 LOADER KONFIGURATION
# Hier alle Parameter fuer den ERA5 Loader anpassen
# ============================================================
ERA5_CONFIG = {
    "file":      ERA5_FILE,
    "format":    "netcdf",
    "variables": ["u10", "v10"],   # Windkomponenten
    "combine":   "wind_speed",     # sqrt(u² + v²)
    "lat":       35.5,             # Gitterpunkt Latitude
    "lon":       -97.5,            # Gitterpunkt Longitude
    "units":     "m/s",
    # Fuer andere Variablen z.B. nur Temperatur:
    # "variables": ["t2m"],
    # "combine":   "single",
}

SYSTEMS = [
    {"name": "kic_6922244",   "file": KIC6922244_FILE, "loader": "load_kic_time_flux"},
    # ERA5 Wind aktivieren:
    # {"name": "era5_wind", "file": ERA5_FILE, "loader": "load_era5_wind"},
]

# Alle verfuegbaren Systeme (auskommentiert):
# SYSTEMS = [
#     {"name": "solar",         "file": SOLAR_FILE,      "loader": "load_solar_daily"},
#     {"name": "nasdaq",        "file": NASDAQ_FILE,     "loader": "load_nasdaq_series"},
#     {"name": "nasdaq_month",  "file": NASDAQ_FOLDER,   "loader": "load_nasdaq_multiday"},
#     {"name": "kic_6922710",   "file": KIC_FILE,        "loader": "load_kic_csv"},
#     {"name": "era5_wind",     "file": ERA5_FILE,       "loader": "load_era5_wind"},
#     {"name": "kic_6922244",   "file": KIC6922244_FILE, "loader": "load_kic_time_flux"},
# ]
FRACTAL_LEVELS = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]  # für den Test überschaubar halten
# FRACTAL_LEVELS = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768] für NQ Börse
# FRACTAL_LEVELS = [0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8] für ERA5
# FRACTAL_LEVELS = [4, 8, 16, 32, 64, 128, 256, 512] für Solar (SN..txt)
MACD_FAST = 6
MACD_SLOW = 26

OUTPUT_DIR = BASE_DIR / "output"
DEBUG = True

# ============================================================
# STORAGE FORMAT: "parquet" (schnell, klein) oder "csv" (lesbar)
# Fuer grosse Daten (1 Jahr): parquet empfohlen
# Fuer kleine Daten (14 Tage): csv fuer einfache Analyse
# Auto-Erkennung beim Laden: nimmt was vorhanden ist
# ============================================================
STORAGE_FORMAT = "parquet"  # umstellen auf "csv" fuer kleine Daten