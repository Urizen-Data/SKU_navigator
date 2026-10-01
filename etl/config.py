"""Пути, периоды, пороги и контракт очистки."""

from pathlib import Path
import pandas as pd

# ─────────────────────────────────────────────────────
# ПОИСК RAW ВВЕРХ ПО ДЕРЕВУ
# ─────────────────────────────────────────────────────
ETL_DIR = Path(__file__).resolve().parent


def find_raw_dir(start: Path, max_up: int = 3) -> Path:
    """Найти папку raw вверх от start включительно."""
    candidates = [start] + [start.parents[i] for i in range(max_up)]
    for c in candidates:
        if (c / "raw").is_dir():
            return c / "raw"
    raise FileNotFoundError(
        f"Папка raw не найдена. Проверены: "
        f"{[str(c / 'raw') for c in candidates]}"
    )


RAW_DIR     = find_raw_dir(ETL_DIR)
RAW_XLSX    = RAW_DIR / "online_retail_II.xlsx"
RAW_PARQUET = RAW_DIR / "online_retail_II.parquet"


RAW_DIR     = find_raw_dir(ETL_DIR)
RAW_XLSX    = RAW_DIR / "online_retail_II.xlsx"
RAW_PARQUET = RAW_DIR / "online_retail_II.parquet"

# ─────────────────────────────────────────────────────
# ВЫХОДНАЯ ПАПКА
# ─────────────────────────────────────────────────────
PROCESSED = ETL_DIR / "data" / "processed"
PROCESSED.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────
# ПЕРИОДЫ
# ─────────────────────────────────────────────────────
P1_START = pd.Timestamp("2009-12-01")
P2_START = pd.Timestamp("2010-12-01")
P2_END   = pd.Timestamp("2011-12-01")

PERIODS = {
    "2009/10": (P1_START, P2_START),
    "2010/11": (P2_START, P2_END),
}

PERIOD_MONTHS = {
    "2009/10": pd.period_range("2009-12", "2010-11", freq="M"),
    "2010/11": pd.period_range("2010-12", "2011-11", freq="M"),
}

ASSESS_DATES = {
    "2009/10": pd.Timestamp("2010-12-01"),
    "2010/11": pd.Timestamp("2011-12-01"),
}

# ─────────────────────────────────────────────────────
# ДУБЛИКАТЫ
# ─────────────────────────────────────────────────────
DUP_COLS = [
    "Invoice", "StockCode", "Description", "Quantity",
    "InvoiceDate", "Price", "Customer ID", "Country",
]

# ─────────────────────────────────────────────────────
# СЛУЖЕБНЫЕ КОДЫ
# ─────────────────────────────────────────────────────
EXCLUDE_CODES = {
    "M", "DOT", "POST", "AMAZONFEE", "C2", "B",
    "ADJUST", "ADJUST2", "BANK CHARGES", "D",
    "S", "TEST001", "TEST002", "CRUK",
}
EXCLUDE_PREFIXES = ("GIFT_",)

# ─────────────────────────────────────────────────────
# ПОРОГИ КЛАССИФИКАЦИЙ
# ─────────────────────────────────────────────────────
ABC_CUTS = {"A": 80, "B": 95}
XYZ_CUTS = {"X": 0.75, "Y": 1.5}

HISTORY_DAYS     = 90
SEASONAL_SHARE   = 0.50
HIGH_RETURN_PCT  = 50
HIGH_KEY_SHARE   = 80
KEY_SEGMENTS     = ["Champions", "Loyal"]