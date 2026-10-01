"""Reusable UI components, styling, and sidebar filters for Streamlit.

Includes executive KPI cards, responsive layout helpers, and safe multi-attribute filtering.
"""
from typing import Tuple, Dict, Any, Optional
import streamlit as st
import pandas as pd


def inject_custom_css():
    """Inject modern executive banking styling into the Streamlit app."""
    st.markdown(
        """
        <style>
        /* Modern Banking UI Theme */
        .main {
            background-color: #F8FAFC;
        }
        .metric-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 10px;
            padding: 18px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            margin-bottom: 12px;
            transition: transform 0.15s ease-in-out;
        }
        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.08);
        }
        .metric-title {
            color: #64748B;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }
        .metric-value {
            color: #0F172A;
            font-size: 1.8rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .metric-subtext {
            color: #94A3B8;
            font-size: 0.8rem;
            margin-top: 4px;
        }
        .insight-box {
            background-color: #F1F5F9;
            border-left: 4px solid #3B82F6;
            padding: 14px 18px;
            border-radius: 4px;
            margin: 10px 0;
        }
        .insight-danger {
            border-left-color: #EF4444;
            background-color: #FEF2F2;
        }
        .insight-warning {
            border-left-color: #F59E0B;
            background-color: #FFFBEB;
        }
        .disclaimer-badge {
            background-color: #EFF6FF;
            color: #1E40AF;
            border: 1px solid #BFDBFE;
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 0.85rem;
            margin-bottom: 15px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_card(
    title: str,
    value: str,
    subtext: Optional[str] = None,
    delta: Optional[str] = None,
    delta_color: str = "normal",
):
    """Render a styled executive metric card."""
    delta_html = ""
    if delta:
        color = "#10B981" if delta_color == "green" else "#EF4444" if delta_color == "red" else "#64748B"
        delta_html = f"<span style='color: {color}; font-size: 0.85rem; font-weight: 600;'>{delta}</span>"

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtext">{subtext if subtext else ''} {delta_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filters and return the filtered dataframe.

    Includes a 'Reset All Filters' button and safe default handlings.
    """
    st.sidebar.header("🎯 Analytics Filters")

    if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    filtered_df = df.copy()

    # 1. Geography
    geos = sorted(df["Geography"].unique().tolist()) if "Geography" in df.columns else []
    selected_geos = st.sidebar.multiselect("Geography", options=geos, default=geos)
    if selected_geos:
        filtered_df = filtered_df[filtered_df["Geography"].isin(selected_geos)]

    # 2. Gender
    genders = sorted(df["Gender"].unique().tolist()) if "Gender" in df.columns else []
    selected_genders = st.sidebar.multiselect("Gender", options=genders, default=genders)
    if selected_genders:
        filtered_df = filtered_df[filtered_df["Gender"].isin(selected_genders)]

    # 3. Age Group
    if "Age_Group" in df.columns:
        age_groups = ["<30", "30-45", "46-60", "60+"]
        selected_age = st.sidebar.multiselect("Age Group", options=age_groups, default=age_groups)
        if selected_age:
            filtered_df = filtered_df[filtered_df["Age_Group"].isin(selected_age)]

    # 4. Credit Score Band
    if "Credit_Score_Band" in df.columns:
        cs_bands = ["Low", "Medium", "High"]
        selected_cs = st.sidebar.multiselect("Credit Score Band", options=cs_bands, default=cs_bands)
        if selected_cs:
            filtered_df = filtered_df[filtered_df["Credit_Score_Band"].isin(selected_cs)]

    # 5. Tenure Group
    if "Tenure_Group" in df.columns:
        tenure_groups = ["New (0-2)", "Mid-term (3-7)", "Long-term (8-10)"]
        selected_tenure = st.sidebar.multiselect("Tenure Group", options=tenure_groups, default=tenure_groups)
        if selected_tenure:
            filtered_df = filtered_df[filtered_df["Tenure_Group"].isin(selected_tenure)]

    # 6. Balance Segment
    if "Balance_Segment" in df.columns:
        bal_segments = ["Zero Balance", "Low Balance", "Medium Balance", "High Balance"]
        selected_bal = st.sidebar.multiselect("Balance Segment", options=bal_segments, default=bal_segments)
        if selected_bal:
            filtered_df = filtered_df[filtered_df["Balance_Segment"].isin(selected_bal)]

    # 7. Salary Segment
    if "Salary_Segment" in df.columns:
        sal_segments = ["Low Salary", "Medium Salary", "High Salary"]
        selected_sal = st.sidebar.multiselect("Salary Segment", options=sal_segments, default=sal_segments)
        if selected_sal:
            filtered_df = filtered_df[filtered_df["Salary_Segment"].isin(selected_sal)]

    # 8. Product Segment
    if "Product_Segment" in df.columns:
        prod_segments = ["1 Product", "2 Products", "3+ Products"]
        selected_prod = st.sidebar.multiselect("Product Count", options=prod_segments, default=prod_segments)
        if selected_prod:
            filtered_df = filtered_df[filtered_df["Product_Segment"].isin(selected_prod)]

    # 9. Active Membership
    act_choice = st.sidebar.radio(
        "Activity Status", options=["All", "Active Members Only", "Inactive Members Only"], index=0
    )
    if act_choice == "Active Members Only":
        filtered_df = filtered_df[filtered_df["IsActiveMember"] == 1]
    elif act_choice == "Inactive Members Only":
        filtered_df = filtered_df[filtered_df["IsActiveMember"] == 0]

    # 10. Credit Card Ownership
    cc_choice = st.sidebar.radio(
        "Credit Card Holder", options=["All", "Has Card Only", "No Card Only"], index=0
    )
    if cc_choice == "Has Card Only":
        filtered_df = filtered_df[filtered_df["HasCrCard"] == 1]
    elif cc_choice == "No Card Only":
        filtered_df = filtered_df[filtered_df["HasCrCard"] == 0]

    # 11. Churn Status
    churn_filter = st.sidebar.radio(
        "Customer Churn Status", options=["All", "Retained Customers Only", "Churned Customers Only"], index=0
    )
    if churn_filter == "Retained Customers Only":
        filtered_df = filtered_df[filtered_df["Exited"] == 0]
    elif churn_filter == "Churned Customers Only":
        filtered_df = filtered_df[filtered_df["Exited"] == 1]

    # Display active sample size in sidebar
    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"📊 **Active Sample:** {len(filtered_df):,} / {len(df):,} accounts "
        f"({(len(filtered_df)/len(df)*100):.1f}%)"
    )

    return filtered_df


def render_csv_download_button(df: pd.DataFrame, filename: str, label: str = "📥 Download Filtered Data (CSV)"):
    """Provide clean CSV download button without sensitive information."""
    export_df = df.copy()
    if "Surname" in export_df.columns:
        export_df = export_df.drop(columns=["Surname"])
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
    )
