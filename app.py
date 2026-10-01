"""European Banking Customer Churn & Segmentation Analytics Application.

Professional Streamlit application for retail banking executive intelligence,
demographic risk modeling, and high-value customer balance exposure analytics.
"""
import streamlit as st
import pandas as pd
import numpy as np

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="European Banking Churn Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.data_loader import load_cleaned_data
from src.kpi_calculations import (
    calculate_overall_kpis,
    calculate_high_value_kpis,
    calculate_segment_kpis,
)
from src.segmentation import (
    analyze_univariate_segment,
    generate_churn_pivot_table,
)
from src.churn_analysis import (
    compare_cohorts,
    run_statistical_tests,
    train_baseline_churn_models,
)
from src.insights import generate_executive_insights
from src.feature_engineering import get_segmentation_thresholds
from dashboard.components import (
    inject_custom_css,
    render_kpi_card,
    render_sidebar_filters,
    render_csv_download_button,
)
from dashboard.charts import (
    plot_churn_donut,
    plot_churn_rate_by_dimension,
    plot_country_comparison_bar,
    plot_heatmap_churn_matrix,
    plot_distribution_comparison,
    plot_bubble_segment_risk,
    create_empty_figure,
)


@st.cache_data(show_spinner=False)
def get_cached_data() -> pd.DataFrame:
    """Load and cache the processed European Banking dataset."""
    return load_cleaned_data()


def main():
    # Inject Custom Banking Styles
    inject_custom_css()

    # Load Data
    try:
        raw_df = get_cached_data()
    except Exception as e:
        st.error(f"⚠️ Error loading dataset: {e}")
        st.info("Please verify that data exists in `data/raw/European_Bank.csv` or `data/European_Bank.csv`.")
        return

    # Sidebar Navigation & Filtering
    st.sidebar.title("🏦 European Banking BI")
    nav_page = st.sidebar.selectbox(
        "📍 Navigation View",
        [
            "1. Executive Overview",
            "2. Customer Segmentation",
            "3. Geography Analysis",
            "4. Churn Drivers & Statistics",
            "5. High-Value Churn Explorer",
        ],
        index=0,
    )

    st.sidebar.markdown("---")
    filtered_df = render_sidebar_filters(raw_df)

    # Calculate Core Dynamic Metrics
    overall_kpis = calculate_overall_kpis(filtered_df)
    hv_kpis = calculate_high_value_kpis(filtered_df)
    thresholds = get_segmentation_thresholds(raw_df)

    # Zero Data Guardrail
    if len(filtered_df) == 0:
        st.warning("⚠️ No customer records match the selected sidebar filters. Please adjust or reset filters.")
        st.stop()

    # =========================================================================
    # PAGE 1: EXECUTIVE OVERVIEW
    # =========================================================================
    if nav_page == "1. Executive Overview":
        st.title("European Banking Customer Churn Analytics")
        st.markdown(
            "Portfolio health monitoring, customer retention trends, and dynamic risk diagnostics across European markets."
        )

        # KPI Row
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            render_kpi_card("Total Customers", f"{overall_kpis['total_customers']:,}")
        with col2:
            render_kpi_card(
                "Churned Customers",
                f"{overall_kpis['churned_customers']:,}",
                delta=f"{overall_kpis['churn_rate']:.1f}% Churn",
                delta_color="red",
            )
        with col3:
            render_kpi_card(
                "Retention Rate",
                f"{overall_kpis['retention_rate']:.1f}%",
                subtext=f"{overall_kpis['retained_customers']:,} Active Accounts",
                delta_color="green",
            )
        with col4:
            render_kpi_card("Average Balance", f"€{overall_kpis['avg_balance']:,.2f}")
        with col5:
            render_kpi_card(
                "High-Value Churn Rate",
                f"{hv_kpis['high_value_churn_rate']:.1f}%",
                subtext=f"{hv_kpis['high_value_churn_count']:,} / {hv_kpis['high_value_customer_count']:,} churned",
            )
        with col6:
            render_kpi_card(
                "Balance Exposure",
                f"€{overall_kpis['total_balance_exposure']:,.0f}",
                subtext="Held by churned accounts",
            )

        st.markdown("---")

        # Visualizations Row
        c1, c2, c3 = st.columns([1, 1.2, 1.2])
        with c1:
            st.plotly_chart(plot_churn_donut(filtered_df), use_container_width=True)
        with c2:
            st.plotly_chart(
                plot_churn_rate_by_dimension(filtered_df, "Geography", "Observed Churn by Country", "Geography"),
                use_container_width=True,
            )
        with c3:
            st.plotly_chart(
                plot_churn_rate_by_dimension(filtered_df, "Age_Group", "Observed Churn by Age Cohort", "Age Group"),
                use_container_width=True,
            )

        # Dynamic Key Insights Section
        st.subheader("💡 Dynamic Key Insights & Automated Findings")
        insights = generate_executive_insights(filtered_df)
        for ins in insights:
            box_class = (
                "insight-danger"
                if ins["type"] == "danger"
                else "insight-warning"
                if ins["type"] == "warning"
                else "insight-box"
            )
            st.markdown(
                f"""
                <div class="insight-box {box_class}">
                    <strong>{ins['category']}: {ins['title']}</strong><br>
                    <span>{ins['detail']}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # =========================================================================
    # PAGE 2: CUSTOMER SEGMENTATION
    # =========================================================================
    elif nav_page == "2. Customer Segmentation":
        st.title("👥 Customer Segmentation Analytics")
        st.markdown(
            "Empirical customer segmentation across demographic cohorts, relationship maturity, financial tiers, and product holdings."
        )

        tabs = st.tabs([
            "Age Group",
            "Credit Score Band",
            "Tenure Group",
            "Balance Segment",
            "Salary Segment",
            "Product Count",
        ])

        segment_configs = [
            ("Age_Group", "Age Cohort Segmentation", tabs[0]),
            ("Credit_Score_Band", "Credit Score Tier Segmentation", tabs[1]),
            ("Tenure_Group", "Tenure Relationship Maturity", tabs[2]),
            ("Balance_Segment", "Account Balance Segmentation", tabs[3]),
            ("Salary_Segment", "Estimated Salary Segmentation", tabs[4]),
            ("Product_Segment", "Product Holding Segmentation", tabs[5]),
        ]

        for col_name, tab_title, tab_ctx in segment_configs:
            with tab_ctx:
                st.subheader(f"📊 {tab_title}")
                summary_df = analyze_univariate_segment(filtered_df, col_name)

                if summary_df.empty:
                    st.info("No data available for this segment.")
                    continue

                c1, c2 = st.columns([1.2, 1])
                with c1:
                    st.plotly_chart(
                        plot_churn_rate_by_dimension(filtered_df, col_name, f"Churn Rate by {col_name.replace('_', ' ')}"),
                        use_container_width=True,
                    )
                with c2:
                    st.plotly_chart(
                        plot_bubble_segment_risk(filtered_df, col_name, f"Risk & Balance Exposure: {col_name.replace('_', ' ')}"),
                        use_container_width=True,
                    )

                st.markdown("##### Detailed Segmentation Breakdown")
                # Format table display
                display_table = summary_df.copy()
                st.dataframe(
                    display_table.style.format({
                        "Customer_Count": "{:,}",
                        "Churn_Count": "{:,}",
                        "Retained_Count": "{:,}",
                        "Churn_Rate_Pct": "{:.2f}%",
                        "Retention_Rate_Pct": "{:.2f}%",
                        "Population_Share_Pct": "{:.2f}%",
                        "Avg_Balance": "€{:,.2f}",
                        "Avg_Salary": "€{:,.2f}",
                        "Avg_Age": "{:.1f}",
                        "Avg_CreditScore": "{:.1f}",
                        "Active_Rate": "{:.2f}%",
                        "Balance_Exposure": "€{:,.2f}",
                    }),
                    use_container_width=True,
                )

                if any(summary_df["Small_Sample_Flag"]):
                    st.caption("⚠️ Note: Segments with under 50 customers are flagged for cautious statistical interpretation.")

    # =========================================================================
    # PAGE 3: GEOGRAPHY ANALYSIS
    # =========================================================================
    elif nav_page == "3. Geography Analysis":
        st.title("🌍 Geographic Market Analysis")
        st.markdown("Comparative market performance and cross-segment behavioral patterns across France, Germany, and Spain.")

        # Country KPI summary table
        geo_summary = calculate_segment_kpis(filtered_df, "Geography")
        st.markdown("##### Regional Financial & Retention Benchmark")
        st.dataframe(
            geo_summary.style.format({
                "Total_Customers": "{:,}",
                "Retained_Customers": "{:,}",
                "Churned_Customers": "{:,}",
                "Churn_Rate_Pct": "{:.2f}%",
                "Retention_Rate_Pct": "{:.2f}%",
                "Avg_Balance": "€{:,.2f}",
                "Total_Balance_Exposure": "€{:,.2f}",
                "Avg_CreditScore": "{:.1f}",
                "Avg_Age": "{:.1f}",
                "Active_Member_Rate_Pct": "{:.2f}%",
            }),
            use_container_width=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(plot_country_comparison_bar(filtered_df), use_container_width=True)
        with c2:
            st.plotly_chart(
                plot_churn_rate_by_dimension(filtered_df, "Geography", "Churn Rate Comparison by Country"),
                use_container_width=True,
            )

        st.markdown("---")
        st.subheader("🔥 Cross-Dimensional Market Heatmaps")
        h1, h2 = st.columns(2)
        with h1:
            st.plotly_chart(
                plot_heatmap_churn_matrix(
                    filtered_df,
                    "Geography",
                    "Age_Group",
                    "Geography × Age Cohort Churn Rates (%)",
                    "Geography",
                    "Age Group",
                ),
                use_container_width=True,
            )
        with h2:
            st.plotly_chart(
                plot_heatmap_churn_matrix(
                    filtered_df,
                    "Geography",
                    "Balance_Segment",
                    "Geography × Balance Segment Churn Rates (%)",
                    "Geography",
                    "Balance Segment",
                ),
                use_container_width=True,
            )

    # =========================================================================
    # PAGE 4: CHURN DRIVERS & STATISTICS
    # =========================================================================
    elif nav_page == "4. Churn Drivers & Statistics":
        st.title("🔬 Churn Drivers & Statistical Analysis")
        st.markdown(
            """
            <div class="disclaimer-badge">
                ⚖️ <strong>Statistical Disclaimer:</strong> All reported metrics reflect observed statistical associations
                within historical observational data and do not establish direct causality.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("1. Cohort Profile Comparison (Churned vs Retained)")
        cohort_diff = compare_cohorts(filtered_df)
        st.dataframe(
            cohort_diff.style.format({
                "Retained_Mean": "{:,.2f}",
                "Churned_Mean": "{:,.2f}",
                "Mean_Difference": "{:+,.2f}",
                "Pct_Difference": "{:+,.2f}%",
                "Retained_Median": "{:,.2f}",
                "Churned_Median": "{:,.2f}",
                "Retained_Std": "{:,.2f}",
                "Churned_Std": "{:,.2f}",
            }),
            use_container_width=True,
        )

        st.markdown("---")
        st.subheader("2. Distribution Visualizer: Churned vs Retained")
        d1, d2 = st.columns(2)
        with d1:
            st.plotly_chart(
                plot_distribution_comparison(filtered_df, "Age", "Age Distribution by Retention Status", "Customer Age"),
                use_container_width=True,
            )
        with d2:
            st.plotly_chart(
                plot_distribution_comparison(filtered_df, "Balance", "Balance Distribution by Retention Status", "Account Balance (€)"),
                use_container_width=True,
            )

        st.markdown("---")
        st.subheader("3. Formal Hypothesis Testing (Chi-Square & Welch's t-Test)")
        stat_results = run_statistical_tests(filtered_df)
        st.dataframe(
            stat_results,
            use_container_width=True,
        )

        st.markdown("---")
        with st.expander("🤖 Optional Predictive Machine Learning Baseline (Extension)", expanded=False):
            st.markdown(
                "Supervised baseline models (Logistic Regression vs. Random Forest) trained with an 80/20 train/test split. "
                "Predictive identifiers (`CustomerId`, `Surname`) are strictly excluded to avoid data leakage."
            )
            try:
                ml_res = train_baseline_churn_models(raw_df)
                m1, m2 = st.columns(2)
                with m1:
                    st.markdown("#### Logistic Regression Baseline")
                    st.metric("ROC-AUC", f"{ml_res['logistic_regression']['roc_auc']:.4f}")
                    st.metric("Accuracy", f"{ml_res['logistic_regression']['accuracy']:.2f}%")
                    st.metric("F1-Score", f"{ml_res['logistic_regression']['f1']:.2f}%")
                with m2:
                    st.markdown("#### Random Forest Classifier")
                    st.metric("ROC-AUC", f"{ml_res['random_forest']['roc_auc']:.4f}")
                    st.metric("Accuracy", f"{ml_res['random_forest']['accuracy']:.2f}%")
                    st.metric("F1-Score", f"{ml_res['random_forest']['f1']:.2f}%")

                st.markdown("##### Top Predictive Feature Importances (Random Forest)")
                st.dataframe(ml_res["random_forest"]["feature_importances"].style.format({"Importance": "{:.4f}"}), use_container_width=True)
            except Exception as e:
                st.warning(f"Could not compute baseline ML models: {e}")

    # =========================================================================
    # PAGE 5: HIGH-VALUE CHURN EXPLORER
    # =========================================================================
    elif nav_page == "5. High-Value Churn Explorer":
        st.title("💎 High-Value Customer & Balance Exposure Explorer")
        st.markdown(
            f"""
            Empirical identification of high-value banking relationships based on calculated upper financial tiers:
            - **Balance Threshold:** $> €{thresholds['balance_p66_nonzero']:,.2f}$ (66th percentile of non-zero accounts)
            - **Estimated Salary Threshold:** $> €{thresholds['salary_p66']:,.2f}$ (66th percentile of salaries)
            """
        )

        hv_col1, hv_col2, hv_col3, hv_col4 = st.columns(4)
        with hv_col1:
            render_kpi_card("High-Value Customers", f"{hv_kpis['high_value_customer_count']:,}")
        with hv_col2:
            render_kpi_card(
                "High-Value Churned",
                f"{hv_kpis['high_value_churn_count']:,}",
                delta=f"{hv_kpis['high_value_churn_rate']:.1f}% Churn Rate",
                delta_color="red",
            )
        with hv_col3:
            render_kpi_card(
                "Balance Exposure",
                f"€{hv_kpis['high_value_balance_exposure']:,.2f}",
                subtext="Total balance held by churned high-value accounts",
            )
        with hv_col4:
            render_kpi_card(
                "Share of Churners",
                f"{hv_kpis['high_value_churn_ratio']:.1f}%",
                subtext="% of all churned customers in high-value tier",
            )

        st.markdown("---")
        st.subheader("🔍 High-Value Account Registry & Data Export")
        st.caption("Displaying customer financial profiles. Note: Surname is strictly excluded for data privacy.")

        # Show High-Value filter subset
        hv_only = st.checkbox("Show High-Value Accounts Only", value=True)
        if hv_only:
            display_df = filtered_df[filtered_df["Is_High_Value"] == 1]
        else:
            display_df = filtered_df

        columns_to_show = [
            "CustomerId",
            "Geography",
            "Gender",
            "Age",
            "CreditScore",
            "Balance",
            "NumOfProducts",
            "IsActiveMember",
            "EstimatedSalary",
            "Exited",
        ]
        available_cols = [c for c in columns_to_show if c in display_df.columns]

        st.dataframe(
            display_df[available_cols].style.format({
                "CustomerId": "{}",
                "Age": "{:d}",
                "CreditScore": "{:d}",
                "Balance": "€{:,.2f}",
                "EstimatedSalary": "€{:,.2f}",
                "NumOfProducts": "{:d}",
                "IsActiveMember": "{:d}",
                "Exited": "{:d}",
            }),
            use_container_width=True,
            height=400,
        )

        st.markdown("---")
        render_csv_download_button(display_df[available_cols], "high_value_churn_export.csv")


if __name__ == "__main__":
    main()
