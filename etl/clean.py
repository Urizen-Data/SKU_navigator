"""Очистка транзакций по контракту."""

import numpy as np
import pandas as pd

from .config import (
    DUP_COLS, EXCLUDE_CODES, EXCLUDE_PREFIXES,
    P1_START, P2_START, P2_END,
)


def clean_sales(raw: pd.DataFrame) -> pd.DataFrame:
    """Полная очистка. Возвращает продажи и возвраты вместе."""
    rows_before = len(raw)

    # 1) Дубли по 8 полям
    clean = raw.drop_duplicates(subset=DUP_COLS, keep="first").copy()
    dup_removed = rows_before - len(clean)

    # 2) Флаг отмены
    clean["IsCancel"] = (
        clean["Invoice"].astype("string").str.strip().str.upper()
        .str.startswith("C", na=False)
    )

        # 3) Служебные позиции
    #    StockCode нормализуем по регистру и пробелам один раз —
    #    иначе Power BI считает «150-56 BL» и «150-56 bl» дубликатами.
    clean["StockCode"] = (
        clean["StockCode"].astype("string").str.strip().str.upper()
    )
    stock = clean["StockCode"]
    nonproduct = (
        stock.isin(EXCLUDE_CODES)
        | stock.str.startswith(EXCLUDE_PREFIXES, na=False)
    )

    # 4) Допустимые операции: продажа без C, отмена с C
    valid_operation = (
        (~clean["IsCancel"] & clean["Quantity"].gt(0))
        | (clean["IsCancel"] & clean["Quantity"].lt(0))
    )

    # 5) Price > 0
    price_valid = clean["Price"].gt(0)

    # 6) Финальный фильтр
    keep = valid_operation & price_valid & ~nonproduct
    clean = clean.loc[keep].copy()

    # 7) Amount со знаком
    clean["Amount"] = clean["Quantity"] * clean["Price"]

    # 8) Период [P1_START, P2_END)
    clean = clean.loc[
        clean["InvoiceDate"].ge(P1_START)
        & clean["InvoiceDate"].lt(P2_END)
    ].copy()

    clean["Period"] = pd.cut(
        clean["InvoiceDate"],
        bins=pd.to_datetime([P1_START, P2_START, P2_END]),
        labels=["2009/10", "2010/11"],
        right=False,
    )

    # Контроль
    assert clean["Price"].gt(0).all()
    assert clean.loc[~clean["IsCancel"], "Quantity"].gt(0).all()
    assert clean.loc[clean["IsCancel"], "Quantity"].lt(0).all()
    assert clean["Period"].notna().all()

    clean = clean.reset_index(drop=True)

    print(f"  Строк до:      {rows_before:>10,}")
    print(f"  Удалено дублей:{dup_removed:>10,}")
    print(f"  Осталось:      {len(clean):>10,}")
    print(f"    Продажи:     {(~clean['IsCancel']).sum():>10,}")
    print(f"    Возвраты:    {clean['IsCancel'].sum():>10,}")

    return clean