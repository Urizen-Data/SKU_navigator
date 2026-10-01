"""Загрузка raw: Excel → Parquet → DataFrame."""

import pandas as pd

from .config import RAW_XLSX, RAW_PARQUET


def load_raw() -> pd.DataFrame:
    """Загрузить raw. Если parquet есть — читаем его."""
    if not RAW_PARQUET.exists():
        sheets = pd.read_excel(RAW_XLSX, sheet_name=None)
        df = pd.concat(
            [s.assign(source_sheet=name) for name, s in sheets.items()],
            ignore_index=True,
        )

        df["Invoice"]       = df["Invoice"].astype(str)
        df["StockCode"]     = df["StockCode"].astype(str)
        df["Description"]   = df["Description"].astype("string")
        df["Quantity"]      = df["Quantity"].astype("int32")
        df["InvoiceDate"]   = pd.to_datetime(df["InvoiceDate"])
        df["Price"]         = df["Price"].astype("float64")
        df["Customer ID"]   = df["Customer ID"].astype("Int32")
        df["Country"]       = df["Country"].astype("category")
        df["source_sheet"]  = df["source_sheet"].astype("category")

        df.to_parquet(RAW_PARQUET, index=False)
        print(f"  Parquet создан: {RAW_PARQUET}")
    else:
        df = pd.read_parquet(RAW_PARQUET)
        print(f"  Parquet загружен: {RAW_PARQUET}")

    print(f"  Строк: {len(df):,}")
    print(f"  Период: {df['InvoiceDate'].min()} — {df['InvoiceDate'].max()}")

    return df