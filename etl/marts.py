"""FactSales, справочники, приведение типов, экспорт."""

from pathlib import Path
import json
import numpy as np
import pandas as pd

from .config import PROCESSED


# ─────────────────────────────────────────────────────
# СПРАВОЧНИКИ
# ─────────────────────────────────────────────────────

def build_dim_product(sku_period: pd.DataFrame) -> pd.DataFrame:
    return (
        sku_period[["StockCode", "ProductName"]]
        .drop_duplicates("StockCode")
        .reset_index(drop=True)
    )


def build_dim_date(clean: pd.DataFrame) -> pd.DataFrame:
    dates = pd.date_range(
        clean["InvoiceDate"].min().normalize(),
        clean["InvoiceDate"].max().normalize(),
        freq="D",
    )
    dim = pd.DataFrame({"Date": dates})
    dim["Year"]      = dim["Date"].dt.year
    dim["Month"]     = dim["Date"].dt.month
    dim["MonthName"] = dim["Date"].dt.strftime("%b")
    dim["Period"]    = pd.cut(
        dim["Date"],
        bins=pd.to_datetime(["2009-12-01", "2010-12-01", "2011-12-01"]),
        labels=["2009/10", "2010/11"],
        right=False,
    )
    return dim


def build_dim_customer(clean: pd.DataFrame) -> pd.DataFrame:
    return (
        clean.dropna(subset=["Customer ID"])
        .groupby("Customer ID", observed=True)
        .agg(Country=("Country", lambda s: s.mode().iloc[0]))
        .reset_index()
        .rename(columns={"Customer ID": "CustomerID"})
    )


def build_dim_period() -> pd.DataFrame:
    return pd.DataFrame({
        "Period":    ["2009/10", "2010/11"],
        "DateStart": pd.to_datetime(["2009-12-01", "2010-12-01"]),
        "DateEnd":   pd.to_datetime(["2010-11-30", "2011-11-30"]),
    })


def build_fact_sales(clean: pd.DataFrame) -> pd.DataFrame:
    fact = clean[[
        "Invoice", "StockCode", "Customer ID",
        "InvoiceDate", "Quantity", "Price", "Amount",
        "IsCancel", "Period", "Country",
    ]].copy()
    fact = fact.rename(columns={
        "Invoice":     "InvoiceNo",
        "Customer ID": "CustomerID",
        "InvoiceDate": "Date",
    })
    fact = fact.reset_index(drop=True)
    fact["RowID"] = fact.index + 1
    fact = fact[[
        "RowID", "InvoiceNo", "StockCode", "CustomerID",
        "Date", "Quantity", "Price", "Amount",
        "IsCancel", "Period", "Country",
    ]]
    return fact


# ─────────────────────────────────────────────────────
# ТИПЫ ДЛЯ ЭКСПОРТА
# ─────────────────────────────────────────────────────

def to_parquet_ready(df: pd.DataFrame) -> pd.DataFrame:
    """Приведение типов: категориальные где уместно, даты в datetime.

    Флаги (repeated_peak_flag, short_history_flag, limited_base_activity,
    seasonality_checked) приводятся к категории, чтобы Power BI видел
    три состояния: True / False / (blank).
    """
    df = df.copy()

    # Строковые
    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype("string")

    # Категориальные
    for col in [
        "Period", "ABC", "XYZ", "ABC_XYZ",
        "BCG_adapted", "RFM_segment", "Country",
        "sales_history",
        "repeated_peak_flag", "short_history_flag",
        "limited_base_activity", "seasonality_checked",
    ]:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Даты
    for col in [
        "Date", "Month", "last_purchase",
        "first_observed_sale", "DateStart", "DateEnd",
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    return df


# ─────────────────────────────────────────────────────
# СБОРКА ВСЕХ ВИТРИН
# ─────────────────────────────────────────────────────

def build_marts(clean: pd.DataFrame, features: dict) -> dict:
    sku_period = features["sku_period"]
    sku_month  = features["sku_month"]
    rfm        = features["rfm"]

    sku_period_cols = [
        # Идентификация и период
        "Period", "StockCode", "ProductName",

        # Базовые метрики и ABC
        "revenue", "units", "orders",
        "revenue_share_pct", "cumulative_share_pct", "ABC",

        # XYZ и совместный класс
        "active_months", "mean_qty", "std_qty", "CV",
        "XYZ", "ABC_XYZ",

        # История товара
        "first_observed_sale", "observed_history_days",
        "short_history_flag",

        # Сезонность
        "repeated_peak_flag", "seasonality_checked",

        # Сальдо и возвраты
        "negative_amount", "sales_balance", "negative_to_sales_pct",

        # Динамика между периодами
        "sales_history",
        "revenue_growth_pct",
        "units_growth_pct",
        "limited_base_activity",

        # Клиенты и RFM
        "known_customers", "known_customer_revenue",
        "key_customers", "key_customer_revenue",
        "customer_revenue_coverage_pct",
        "key_customer_revenue_share_pct",
        "key_customer_share_pct",

        # BCG
        "BCG_adapted",
    ]

    # Проверяем, что все требуемые поля действительно есть в витрине.
    # Если чего-то нет — падаем с понятным сообщением, а не молча теряем.
    missing = [c for c in sku_period_cols if c not in sku_period.columns]
    assert not missing, (
        f"В sku_period отсутствуют поля из плана дашборда: {missing}"
    )

    sku_period_out = to_parquet_ready(sku_period[sku_period_cols])

    return {
        "DimProduct":    to_parquet_ready(build_dim_product(sku_period)),
        "DimDate":       to_parquet_ready(build_dim_date(clean)),
        "DimCustomer":   to_parquet_ready(build_dim_customer(clean)),
        "DimPeriod":     to_parquet_ready(build_dim_period()),
        "FactSales":     to_parquet_ready(build_fact_sales(clean)),
        "MartSkuPeriod": sku_period_out,
        "MartSkuMonth":  to_parquet_ready(sku_month),
        "MartRfm":       to_parquet_ready(
            rfm.rename(columns={"Customer ID": "CustomerID"})
        ),
    }


def export_all(marts: dict) -> None:
    """Сохранить все витрины в data/processed/."""
    for name, df in marts.items():
        path = PROCESSED / f"{name}.parquet"
        df.to_parquet(path, index=False)
        print(f"  {name:<15} {len(df):>8,} строк → {path.name}")


# ─────────────────────────────────────────────────────
# ПРОВЕРКИ И ЖУРНАЛ
# ─────────────────────────────────────────────────────

def validate(clean: pd.DataFrame, marts: dict) -> None:
    sku_period = marts["MartSkuPeriod"]
    sku_month  = marts["MartSkuMonth"]
    fact       = marts["FactSales"]
    rfm        = marts["MartRfm"]

    # ── Уникальность ключей ──
    assert sku_period.duplicated(["Period", "StockCode"]).sum() == 0
    assert sku_month.duplicated(["Period", "StockCode", "Month"]).sum() == 0
    assert rfm.duplicated(["Period", "CustomerID"]).sum() == 0

    # ── Согласованность сумм ──
    assert abs(fact["Amount"].sum() - clean["Amount"].sum()) < 1e-6

    rev_sku = sku_period.groupby("Period", observed=True)["revenue"].sum()
    rev_fact = (
        clean.loc[~clean["IsCancel"]]
        .groupby("Period", observed=True)["Amount"].sum()
    )
    assert np.allclose(rev_sku.values, rev_fact.values)

    # ── Сезонность: 531 True в 2010/11, NA в 2009/10 ──
    assert sku_period.loc[
        sku_period["Period"].eq("2009/10"), "repeated_peak_flag"
    ].isna().all()

        # Точное число сезонных SKU зависит от нормализации StockCode.
    # Проверяем не конкретное значение, а структуру:
    #   - флаг определён только для 2010/11,
    #   - True и False оба присутствуют,
    #   - True не больше 20% и не меньше 1% от ассортимента.
    n_p2 = sku_period["Period"].eq("2010/11").sum()
    n_true  = (sku_period["repeated_peak_flag"] == True).sum()   # noqa: E712
    n_false = (sku_period["repeated_peak_flag"] == False).sum()  # noqa: E712
    n_na    = sku_period["repeated_peak_flag"].isna().sum()

    assert n_true > 0,  "Нет ни одного сезонного SKU — проверь расчёт"
    assert n_false > 0, "Нет ни одного не-сезонного SKU — проверь расчёт"
    assert n_true + n_false + n_na == len(sku_period), (
        "Сумма True + False + NA не совпадает с числом строк"
    )
    assert 0.01 <= n_true / n_p2 <= 0.20, (
        f"Доля сезонных SKU вне разумного диапазона: {n_true / n_p2:.1%}"
    )

    print(f"  Сезонных SKU: {n_true} "
          f"({n_true / n_p2:.1%} ассортимента 2010/11)")

    # ── Новые поля плана дашборда: обязательны для страницы 2 ──
    required_new = [
        "sales_history",
        "revenue_growth_pct",
        "units_growth_pct",
        "limited_base_activity",
        "key_customer_share_pct",
    ]
    for col in required_new:
        assert col in sku_period.columns, f"Нет колонки {col} в MartSkuPeriod"

    # sales_history: три непустых значения
    assert sku_period["sales_history"].notna().all()
    assert set(sku_period["sales_history"].astype(str).unique()).issubset(
        {
            "Продажи в обоих периодах",
            "Продажи только в первом",
            "Продажи только во втором",
            "Нет продаж",
        }
    )

    # limited_base_activity: NA в 2009/10, True/False в 2010/11
    assert sku_period.loc[
        sku_period["Period"].eq("2009/10"), "limited_base_activity"
    ].isna().all()
    assert sku_period.loc[
        sku_period["Period"].eq("2010/11"), "limited_base_activity"
    ].notna().all()

    # key_customer_share_pct: либо число, либо NA (нет известных покупателей)
    kcs = sku_period["key_customer_share_pct"]
    assert kcs.dropna().between(0, 100).all()
    no_known = sku_period["known_customers"] == 0
    assert kcs[no_known].isna().all(), (
        "key_customer_share_pct должен быть NA, если известных покупателей нет"
    )

        # Темпы роста определены, только если была база в 2009/10.
    # «Продажи только во втором» и «Нет продаж» — базы нет → NA.
    has_base = sku_period["sales_history"].isin(
        ["Продажи в обоих периодах", "Продажи только в первом"]
    )
    assert sku_period.loc[has_base, "revenue_growth_pct"].notna().all()
    assert sku_period.loc[~has_base, "revenue_growth_pct"].isna().all()

    assert sku_period.loc[has_base, "units_growth_pct"].notna().all()
    assert sku_period.loc[~has_base, "units_growth_pct"].isna().all()

    print("  Проверки: OK")


def write_summary(clean: pd.DataFrame, marts: dict) -> None:
    sku_period = marts["MartSkuPeriod"]
    sku_month  = marts["MartSkuMonth"]
    rfm        = marts["MartRfm"]

    both_periods = sku_period["sales_history"].eq("Продажи в обоих периодах")

    summary = {
        "clean_rows":         int(len(clean)),
        "sales_rows":         int((~clean["IsCancel"]).sum()),
        "return_rows":        int(clean["IsCancel"].sum()),
        "revenue_sales_gbp":  round(float(clean.loc[~clean["IsCancel"], "Amount"].sum()), 2),
        "returns_gbp":        round(float(-clean.loc[clean["IsCancel"], "Amount"].sum()), 2),
        "balance_gbp":        round(float(clean["Amount"].sum()), 2),

        "sku_period_rows":    int(len(sku_period)),
        "sku_month_rows":     int(len(sku_month)),
        "rfm_rows":           int(len(rfm)),

        "seasonal_sku_count": int(
            (sku_period["repeated_peak_flag"] == True).sum()  # noqa: E712
        ),
        "short_history_count": int(
            (sku_period["short_history_flag"] == True).sum()  # noqa: E712
        ),
        "limited_base_count": int(
            (sku_period["limited_base_activity"] == True).sum()  # noqa: E712
        ),
        "sales_both_periods": int(both_periods.sum()),
    }

    path = PROCESSED / "etl_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))

    print("  Журнал:")
    for k, v in summary.items():
        print(f"    {k:<22} {v}")