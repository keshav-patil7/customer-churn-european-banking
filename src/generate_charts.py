"""Script to export static portfolio charts for reports and assets."""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.data_loader import load_cleaned_data


def generate_all_static_charts():
    df = load_cleaned_data()
    os.makedirs("assets/charts", exist_ok=True)
    sns.set_theme(style="whitegrid")

    overall_churn = df["Exited"].mean() * 100.0

    # 1. Churn Donut
    plt.figure(figsize=(6, 6))
    churn_counts = df["Exited"].value_counts()
    plt.pie(
        churn_counts,
        labels=["Retained", "Churned"],
        autopct="%1.1f%%",
        colors=["#10B981", "#EF4444"],
        startangle=90,
        explode=(0, 0.05),
    )
    plt.title("Customer Churn vs Retention Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("assets/charts/churn_distribution_donut.png", dpi=300)
    plt.close()

    # 2. Churn by Geography
    plt.figure(figsize=(8, 5))
    geo_churn = df.groupby("Geography")["Exited"].mean() * 100
    sns.barplot(x=geo_churn.index, y=geo_churn.values, hue=geo_churn.index, legend=False, palette=["#3B82F6", "#EF4444", "#10B981"])
    plt.axhline(overall_churn, color="gray", linestyle="--", label=f"Benchmark ({overall_churn:.1f}%)")
    plt.title("Observed Churn Rate by Geography (%)", fontsize=13, fontweight="bold")
    plt.ylabel("Churn Rate (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/charts/churn_by_geography.png", dpi=300)
    plt.close()

    # 3. Churn by Age Group
    plt.figure(figsize=(8, 5))
    age_churn = df.groupby("Age_Group", observed=False)["Exited"].mean() * 100
    sns.barplot(x=age_churn.index, y=age_churn.values, hue=age_churn.index, legend=False, palette="Blues_r")
    plt.axhline(overall_churn, color="gray", linestyle="--", label=f"Benchmark ({overall_churn:.1f}%)")
    plt.title("Observed Churn Rate by Age Cohort (%)", fontsize=13, fontweight="bold")
    plt.ylabel("Churn Rate (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/charts/churn_by_age_group.png", dpi=300)
    plt.close()

    # 4. Age Distribution
    plt.figure(figsize=(9, 5))
    sns.kdeplot(
        data=df,
        x="Age",
        hue="Churn_Label",
        common_norm=False,
        fill=True,
        palette={"Retained": "#10B981", "Churned": "#EF4444"},
    )
    plt.title("Age Density Distribution: Churned vs Retained", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("assets/charts/age_density_distribution.png", dpi=300)
    plt.close()

    # 5. Balance Distribution
    plt.figure(figsize=(9, 5))
    sns.kdeplot(
        data=df[df["Balance"] > 0],
        x="Balance",
        hue="Churn_Label",
        common_norm=False,
        fill=True,
        palette={"Retained": "#10B981", "Churned": "#EF4444"},
    )
    plt.title("Non-Zero Balance Density Distribution: Churned vs Retained", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("assets/charts/balance_density_distribution.png", dpi=300)
    plt.close()

    print("Static charts generated successfully in assets/charts/")


if __name__ == "__main__":
    generate_all_static_charts()
