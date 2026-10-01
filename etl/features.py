"""Аналитические витрины: sku_period, sku_month, rfm, BCG.

Логика согласована с ноутбуком «Навигатор»:
- ABC/XYZ — по положительным продажам;
- сезонные пики — по каждому периоду отдельно + реальное пересечение;
- RFM — по известным покупателям, границы баллов внутри периода;
- сальдо — выручка продаж минус модуль возвратов;
- sales_history, growth_pct, limited_base_activity — из годового сравнения.
"""

import numpy as np
import pandas as pd

from .config import (
    ABC_CUTS, XYZ_CUTS, PERIOD_MONTHS, ASSESS_DATES,
    HISTORY_DAYS, KEY_SEGMENTS,
)


def build_features(clean: pd.DataFrame) -> dict:
    """Все аналитические витрины за один проход."""
    sales = clean.loc[~clean["IsCancel"]].copy()

    # ─────────────────────────────────────────────
    # 1) SKU × период: базовые метрики и ABC
    # ─────────────────────────────────────────────
    sku = (
        sales.groupby(["Period", "StockCode"], observed=True)
        .agg(
            ProductName=("Description", lambda s: s.mode().iloc[0]
                         if not s.mode().empty else "Без описания"),
            revenue=("Amount", "sum"),
            units=("Quantity", "sum"),
            orders=("Invoice", "nunique"),
        )
        .reset_index()
        .sort_values(["Period", "revenue", "StockCode"],
                     ascending=[True, False, True])
        .reset_index(drop=True)
    )

    sku["revenue_share_pct"] = (
        100 * sku["revenue"]
        / sku.groupby("Period")["revenue"].transform("sum")
    )
    sku["cumulative_share_pct"] = (
        sku.groupby("Period")["revenue_share_pct"].cumsum()
    )
    sku["ABC"] = np.select(
        [sku["cumulative_share_pct"] <= ABC_CUTS["A"],
         sku["cumulative_share_pct"] <= ABC_CUTS["B"]],
        ["A", "B"], default="C"
    )

    # ─────────────────────────────────────────────
    # 2) SKU × месяц: продажи + возвраты (полный ряд)
    # ─────────────────────────────────────────────
    def _month_side(df: pd.DataFrame, sales_side: bool) -> pd.DataFrame:
        sub = df.loc[~df["IsCancel"]] if sales_side else df.loc[df["IsCancel"]]
        parts = []
        for period, month_range in PERIOD_MONTHS.items():
            sp = sub.loc[sub["Period"].eq(period)]
            skus_p = sku.loc[sku["Period"].eq(period), "StockCode"]
            grid = pd.MultiIndex.from_product(
                [skus_p, month_range], names=["StockCode", "Month"]
            )
            agg = (
                sp.assign(Month=sp["InvoiceDate"].dt.to_period("M"))
                .groupby(["StockCode", "Month"], observed=True)
                .agg(qty=("Quantity", "sum"), amount=("Amount", "sum"))
                .reindex(grid, fill_value=0)
                .reset_index()
                .assign(Period=period)
            )
            parts.append(agg)
        return pd.concat(parts, ignore_index=True)

    sku_month = (
        _month_side(clean, True)
        .rename(columns={"qty": "qty_sold", "amount": "revenue_sold"})
        .merge(
            _month_side(clean, False)
            .rename(columns={"qty": "qty_returned", "amount": "amount_returned"}),
            on=["Period", "StockCode", "Month"], how="outer",
            validate="one_to_one",
        )
        .fillna(0)
    )
    sku_month["qty_returned"]    = sku_month["qty_returned"].abs()
    sku_month["amount_returned"] = sku_month["amount_returned"].abs()
    sku_month["revenue_net"]     = (
        sku_month["revenue_sold"] - sku_month["amount_returned"]
    )
    sku_month["Month"] = sku_month["Month"].dt.to_timestamp()

    # ─────────────────────────────────────────────
    # 3) XYZ: полный 12-месячный ряд
    # ─────────────────────────────────────────────
    sku_month_sales = (
        sku_month[["Period", "StockCode", "Month", "qty_sold"]]
        .rename(columns={"qty_sold": "units"})
    )
    xyz = (
        sku_month_sales
        .groupby(["Period", "StockCode"], observed=True)
        .agg(
            active_months=("units", lambda x: x.gt(0).sum()),
            mean_qty=("units", "mean"),
            std_qty=("units", lambda x: x.std(ddof=0)),
        )
        .reset_index()
    )
    xyz["CV"] = xyz["std_qty"] / xyz["mean_qty"]
    xyz["XYZ"] = np.select(
        [xyz["CV"] <= XYZ_CUTS["X"], xyz["CV"] <= XYZ_CUTS["Y"]],
        ["X", "Y"], default="Z"
    )

    sku = sku.merge(
        xyz[["Period", "StockCode", "active_months",
             "mean_qty", "std_qty", "CV", "XYZ"]],
        on=["Period", "StockCode"], how="left", validate="one_to_one"
    )
    sku["ABC_XYZ"] = sku["ABC"] + sku["XYZ"]

    # ─────────────────────────────────────────────
    # 4) BCG между периодами + ростовые метрики
    # ─────────────────────────────────────────────
    rev_pivot = (
        sku.pivot(index="StockCode", columns="Period", values="revenue")
        .rename(columns={"2009/10": "revenue_p1", "2010/11": "revenue_p2"})
        .fillna(0)
    )
    rev_pivot["growth_pct"] = np.where(
        rev_pivot["revenue_p1"] > 0,
        (rev_pivot["revenue_p2"] / rev_pivot["revenue_p1"] - 1) * 100,
        np.nan,
    )
    rev_pivot["revenue_change"] = (
        rev_pivot["revenue_p2"] - rev_pivot["revenue_p1"]
    )

    abc_p2 = (
        sku.loc[sku["Period"].eq("2010/11"), ["StockCode", "ABC"]]
        .rename(columns={"ABC": "ABC_p2"})
        .set_index("StockCode")
    )
    bcg = rev_pivot.join(abc_p2, how="left")
    hs = bcg["ABC_p2"].eq("A")
    ch = bcg["revenue_change"]
    bcg["BCG_adapted"] = np.select(
        [hs & ch.gt(0), hs & ch.lt(0),
         ~hs & ch.gt(0), ~hs & ch.lt(0)],
        ["A — рост", "A — снижение",
         "B/C — рост", "B/C — снижение"],
        default="Без изменения",
    )
    bcg["growth_pct"] = bcg["growth_pct"].fillna(0)

    sku["BCG_adapted"] = sku["StockCode"].map(bcg["BCG_adapted"])

    # 4.1) Признак истории продаж между периодами
    units_pivot = (
        sku.pivot(index="StockCode", columns="Period", values="units")
        .rename(columns={"2009/10": "units_p1", "2010/11": "units_p2"})
        .fillna(0)
    )
    has_p1 = units_pivot["units_p1"].gt(0)
    has_p2 = units_pivot["units_p2"].gt(0)

    sales_history_map = pd.Series(
        np.select(
            [has_p1 & has_p2, ~has_p1 & has_p2, has_p1 & ~has_p2],
            ["Продажи в обоих периодах",
             "Продажи только во втором",
             "Продажи только в первом"],
            default="Нет продаж",
        ),
        index=units_pivot.index,
        name="sales_history",
    )
    sku["sales_history"] = sku["StockCode"].map(sales_history_map)

        # 4.2) Темп роста выручки — только для SKU с базой в первом периоде.
    #      Для SKU без продаж в 2009/10 (только во втором) — NaN.
    rev_pivot_g = (
        sku.pivot(index="StockCode", columns="Period", values="revenue")
        .rename(columns={"2009/10": "rev_p1", "2010/11": "rev_p2"})
        .fillna(0)
    )
    rev_pivot_g["revenue_growth_pct"] = np.where(
        rev_pivot_g["rev_p1"] > 0,
        (rev_pivot_g["rev_p2"] / rev_pivot_g["rev_p1"] - 1) * 100,
        np.nan,
    )
    sku["revenue_growth_pct"] = sku["StockCode"].map(
        rev_pivot_g["revenue_growth_pct"]
    )

    # 4.3) Темп роста количества — аналогично выручке
    units_pivot["units_growth_pct"] = np.where(
        units_pivot["units_p1"] > 0,
        (units_pivot["units_p2"] / units_pivot["units_p1"] - 1) * 100,
        np.nan,
    )
    sku["units_growth_pct"] = sku["StockCode"].map(
        units_pivot["units_growth_pct"]
    )

    # ─────────────────────────────────────────────
    # 5) История товара и сезонные пики
    #    Логика из ноутбука: по каждому периоду отдельно
    #    + реальное пересечение пиковых месяцев между годами.
    # ─────────────────────────────────────────────
    first_sale = sales.groupby("StockCode", observed=True)["InvoiceDate"].min()
    sku["first_observed_sale"] = sku["StockCode"].map(first_sale)
    eval_date = (
        sku["Period"].astype("string").map(ASSESS_DATES).pipe(pd.to_datetime)
    )
    sku["observed_history_days"] = (
        eval_date - sku["first_observed_sale"].dt.normalize()
    ).dt.days
    sku["short_history_flag"] = (
        sku["observed_history_days"].le(HISTORY_DAYS).astype("boolean")
    )

    # 5.1) Признак ограниченной активности базового периода.
    #      Берём active_months за 2009/10 и переносим на обе строки SKU.
    #      Для 2009/10 оставляем NA — сам с собой не сравнивается.
    active_p1 = (
        sku.loc[sku["Period"].eq("2009/10"), ["StockCode", "active_months"]]
        .rename(columns={"active_months": "active_months_p1"})
    )
    sku = sku.merge(active_p1, on="StockCode", how="left")
    sku["limited_base_activity"] = (
        sku["active_months_p1"].le(2)
        .where(sku["Period"].eq("2010/11"))
        .astype("boolean")
    )
    sku = sku.drop(columns=["active_months_p1"])

    # 5.2) Полные месячные ряды: (Period, StockCode) × календарный месяц 1–12
    monthly_by_period = (
        sku_month
        .assign(calendar_month=sku_month["Month"].dt.month)
        .pivot_table(
            index=["Period", "StockCode"],
            columns="calendar_month",
            values="qty_sold",
            aggfunc="sum",
            fill_value=0,
            observed=True,
        )
        .reindex(columns=range(1, 13), fill_value=0)
        .fillna(0)
    )

    # 5.3) Разделяем по периодам, берём только сопоставимые SKU
    p1 = monthly_by_period.xs("2009/10", level="Period")
    p2 = monthly_by_period.xs("2010/11", level="Period")
    common = p1.index.intersection(p2.index)
    p1 = p1.loc[common]
    p2 = p2.loc[common]

    def _peaks(row: pd.Series) -> set:
        return set(row[row > 0].nlargest(3).index)

    share1 = p1.apply(
        lambda x: x.nlargest(3).sum() / x.sum() if x.sum() > 0 else 0,
        axis=1,
    )
    share2 = p2.apply(
        lambda x: x.nlargest(3).sum() / x.sum() if x.sum() > 0 else 0,
        axis=1,
    )
    overlap = pd.Series(
        [len(_peaks(p1.loc[s]) & _peaks(p2.loc[s])) for s in common],
        index=common,
    )

    # 5.4) Итоговый флаг: три условия одновременно
    seasonal_mask = share1.ge(0.5) & share2.ge(0.5) & overlap.ge(2)

    sku["repeated_peak_flag"] = (
        sku["StockCode"]
        .map(seasonal_mask)
        .where(sku["Period"].eq("2010/11"))
        .astype("boolean")
    )
    sku["seasonality_checked"] = sku["repeated_peak_flag"].notna()

    # ─────────────────────────────────────────────
    # 6) Сальдо: продажи − модуль возвратов
    # ─────────────────────────────────────────────
    negative = (
        clean.loc[clean["IsCancel"]]
        .groupby(["Period", "StockCode"], observed=True)["Amount"]
        .sum().abs().rename("negative_amount")
    )
    sku = sku.merge(
        negative, on=["Period", "StockCode"], how="left",
        validate="one_to_one",
    )
    sku["negative_amount"] = sku["negative_amount"].fillna(0)
    sku["sales_balance"] = sku["revenue"] - sku["negative_amount"]
    sku["negative_to_sales_pct"] = (
        100 * sku["negative_amount"] / sku["revenue"]
    )

    # ─────────────────────────────────────────────
    # 7) RFM и клиентские метрики на SKU
    # ─────────────────────────────────────────────
    customer_sales = clean.loc[
        ~clean["IsCancel"] & clean["Customer ID"].notna()
    ].copy()

    parts = []
    for period, date in ASSESS_DATES.items():
        d = customer_sales.loc[customer_sales["Period"].eq(period)]
        t = (
            d.groupby("Customer ID", observed=True)
            .agg(last_purchase=("InvoiceDate", "max"),
                 Frequency=("Invoice", "nunique"),
                 Monetary=("Amount", "sum"))
            .reset_index()
        )
        t["Recency"] = (date - t["last_purchase"].dt.normalize()).dt.days
        t["Period"] = period
        parts.append(t)

    rfm = pd.concat(parts, ignore_index=True)

    for period, g in rfm.groupby("Period", observed=True):
        idx = g.index
        rfm.loc[idx, "R_score"] = pd.qcut(
            g["Recency"].rank(method="first"), 3, labels=[3, 2, 1]
        ).astype(int)
        rfm.loc[idx, "F_score"] = pd.qcut(
            g["Frequency"].rank(method="first"), 3, labels=[1, 2, 3]
        ).astype(int)
        rfm.loc[idx, "M_score"] = pd.qcut(
            g["Monetary"].rank(method="first"), 3, labels=[1, 2, 3]
        ).astype(int)

    r, f, m = rfm["R_score"], rfm["F_score"], rfm["M_score"]
    rfm["RFM_segment"] = np.select(
        [r.eq(3) & f.eq(3) & m.eq(3),
         r.ge(2) & f.ge(2),
         r.eq(3) & f.eq(1),
         r.eq(1) & f.ge(2),
         r.eq(1)],
        ["Champions", "Loyal", "Recent single", "At Risk", "Inactive"],
        default="Regular",
    )

    customer_sku = (
        customer_sales[["Period", "StockCode", "Customer ID", "Amount"]]
        .merge(rfm[["Period", "Customer ID", "RFM_segment"]],
               on=["Period", "Customer ID"], how="left",
               validate="many_to_one")
    )
    key_mask = customer_sku["RFM_segment"].isin(KEY_SEGMENTS)

    # known_customer_revenue — сумма по всем известным покупателям SKU,
    # нужна для customer_revenue_coverage_pct
    sku_client = (
        customer_sku
        .assign(
            known_cust=customer_sku["Customer ID"],
            key_cust=customer_sku["Customer ID"].where(key_mask),
            key_rev=customer_sku["Amount"].where(key_mask, 0),
        )
        .groupby(["Period", "StockCode"], observed=True)
        .agg(
            known_customers=("known_cust", "nunique"),
            known_customer_revenue=("Amount", "sum"),
            key_customers=("key_cust", "nunique"),
            key_customer_revenue=("key_rev", "sum"),
        )
        .reset_index()
    )

    sku = sku.merge(sku_client, on=["Period", "StockCode"],
                    how="left", validate="one_to_one")
    sku[["known_customers", "key_customers"]] = (
        sku[["known_customers", "key_customers"]].fillna(0).astype("int64")
    )
    sku[["known_customer_revenue", "key_customer_revenue"]] = (
        sku[["known_customer_revenue", "key_customer_revenue"]].fillna(0)
    )

    sku["customer_revenue_coverage_pct"] = (
        100 * sku["known_customer_revenue"] / sku["revenue"]
    )
    sku["key_customer_revenue_share_pct"] = (
        100 * sku["key_customer_revenue"] / sku["revenue"]
    )

    # 7.1) Доля ключевых покупателей среди известных покупателей SKU.
    #      Если известных нет — оставляем NA, а не 0.
    sku["key_customer_share_pct"] = (
        100 * sku["key_customers"]
        / sku["known_customers"].replace(0, np.nan)
    )

    return {
        "sku_period": sku,
        "sku_month":  sku_month,
        "rfm":        rfm,
        "bcg":        bcg,
    }