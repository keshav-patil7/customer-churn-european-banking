"""Dynamic narrative insight generation engine.

Calculates evidence-based narrative findings, risk rankings, and sample size cautions
directly from data frames without static hardcoding.
"""
from typing import List, Dict, Any
import pandas as pd
from src.kpi_calculations import calculate_overall_kpis, calculate_high_value_kpis


def generate_executive_insights(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Generate dynamic narrative findings from customer data.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (full or filtered).

    Returns
    -------
    List[Dict[str, Any]]
        List of structured insight dictionaries.
    """
    if len(df) == 0:
        return [{
            "category": "General",
            "type": "warning",
            "title": "No Data Available",
            "detail": "The current filter selection returned 0 customer records.",
        }]

    insights = []
    overall = calculate_overall_kpis(df)
    hv_kpis = calculate_high_value_kpis(df)

    # 1. Geographic Churn Finding
    if "Geography" in df.columns and df["Geography"].nunique() > 1:
        geo_agg = (
            df.groupby("Geography", observed=False)
            .agg(Count=("Exited", "count"), Churn=("Exited", lambda x: (x == 1).sum()))
            .reset_index()
        )
        geo_agg["Rate"] = (geo_agg["Churn"] / geo_agg["Count"] * 100.0).round(2)
        top_geo = geo_agg.sort_values(by="Rate", ascending=False).iloc[0]
        lowest_geo = geo_agg.sort_values(by="Rate", ascending=True).iloc[0]

        insights.append({
            "category": "Geographic Pattern",
            "type": "danger" if top_geo["Rate"] > overall["churn_rate"] else "info",
            "title": f"Elevated Churn in {top_geo['Geography']}",
            "detail": (
                f"{top_geo['Geography']} records the highest observed churn rate at "
                f"{top_geo['Rate']:.2f}% across {top_geo['Count']:,} customers "
                f"({top_geo['Churn']:,} churned). In contrast, {lowest_geo['Geography']} "
                f"exhibits the lowest churn rate at {lowest_geo['Rate']:.2f}% "
                f"across {lowest_geo['Count']:,} customers."
            ),
        })

    # 2. Demographic Age Finding
    if "Age_Group" in df.columns:
        age_agg = (
            df.groupby("Age_Group", observed=False)
            .agg(Count=("Exited", "count"), Churn=("Exited", lambda x: (x == 1).sum()))
            .reset_index()
        )
        age_agg["Rate"] = (age_agg["Churn"] / age_agg["Count"] * 100.0).round(2)
        valid_age = age_agg[age_agg["Count"] >= 30]
        if not valid_age.empty:
            top_age = valid_age.sort_values(by="Rate", ascending=False).iloc[0]
            insights.append({
                "category": "Demographic Pattern",
                "type": "warning",
                "title": f"Highest Churn Cohort: Age Group {top_age['Age_Group']}",
                "detail": (
                    f"Customers in the {top_age['Age_Group']} cohort experience an observed churn rate of "
                    f"{top_age['Rate']:.2f}% ({top_age['Churn']:,} churned out of {top_age['Count']:,} customers), "
                    f"substantially above the portfolio average of {overall['churn_rate']:.2f}%."
                ),
            })

    # 3. Active Membership Impact
    if "IsActiveMember" in df.columns:
        act_agg = (
            df.groupby("IsActiveMember", observed=False)
            .agg(Count=("Exited", "count"), Churn=("Exited", lambda x: (x == 1).sum()))
            .reset_index()
        )
        act_agg["Rate"] = (act_agg["Churn"] / act_agg["Count"] * 100.0).round(2)
        inact_row = act_agg[act_agg["IsActiveMember"] == 0]
        act_row = act_agg[act_agg["IsActiveMember"] == 1]

        if not inact_row.empty and not act_row.empty:
            inact_rate = inact_row["Rate"].values[0]
            act_rate = act_row["Rate"].values[0]
            rate_gap = inact_rate - act_rate
            insights.append({
                "category": "Engagement Pattern",
                "type": "info",
                "title": "Inactivity Associated with Higher Churn",
                "detail": (
                    f"Inactive members exhibit a churn rate of {inact_rate:.2f}% "
                    f"({inact_row['Churn'].values[0]:,}/{inact_row['Count'].values[0]:,}), compared to "
                    f"{act_rate:.2f}% for active members ({act_row['Churn'].values[0]:,}/{act_row['Count'].values[0]:,}), "
                    f"representing an observed risk differential of {rate_gap:+.2f} percentage points."
                ),
            })

    # 4. Product Segment Finding
    if "Product_Segment" in df.columns:
        prod_agg = (
            df.groupby("Product_Segment", observed=False)
            .agg(Count=("Exited", "count"), Churn=("Exited", lambda x: (x == 1).sum()))
            .reset_index()
        )
        prod_agg["Rate"] = (prod_agg["Churn"] / prod_agg["Count"] * 100.0).round(2)
        p3_row = prod_agg[prod_agg["Product_Segment"] == "3+ Products"]

        if not p3_row.empty:
            p3_count = p3_row["Count"].values[0]
            p3_rate = p3_row["Rate"].values[0]
            caution_text = (
                f" (Interpret with caution due to relatively small sample size: {p3_count:,} accounts)"
                if p3_count < 500
                else ""
            )
            insights.append({
                "category": "Product Utilization",
                "type": "danger" if p3_rate > 50 else "warning",
                "title": "High Risk on 3+ Products Accounts",
                "detail": (
                    f"Customers with 3+ products show an extreme observed churn rate of {p3_rate:.2f}% "
                    f"({p3_row['Churn'].values[0]:,}/{p3_count:,}).{caution_text}"
                ),
            })

    # 5. High-Value Financial Balance Exposure
    if hv_kpis["high_value_customer_count"] > 0:
        insights.append({
            "category": "Financial Exposure",
            "type": "danger" if hv_kpis["high_value_churn_rate"] > overall["churn_rate"] else "info",
            "title": f"High-Value Customer Exposure (€{hv_kpis['high_value_balance_exposure']:,.2f})",
            "detail": (
                f"Out of {hv_kpis['high_value_customer_count']:,} identified high-value customers "
                f"(top balance and salary tiers), {hv_kpis['high_value_churn_count']:,} have churned "
                f"({hv_kpis['high_value_churn_rate']:.2f}% churn rate), representing €{hv_kpis['high_value_balance_exposure']:,.2f} "
                f"in balance exposure."
            ),
        })

    return insights
