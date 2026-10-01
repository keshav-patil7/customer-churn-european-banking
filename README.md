# Customer Segmentation & Churn Pattern Analytics in European Banking

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests: Pytest](https://img.shields.io/badge/tests-pytest-green.svg)](tests/)

An enterprise-grade, portfolio-quality data analytics and interactive intelligence dashboard designed to quantify, segment, and model customer attrition across European retail banking markets (France, Germany, Spain).

---

## 📌 Project Overview

Customer churn represents one of the most acute threats to retail banking stability, directly impacting deposit bases, treasury liquidity, and customer lifetime value (LTV). 

Using an audited cohort of **10,000 retail banking accounts** across France, Germany, and Spain, this project delivers:
1. **Dynamic Risk Segmentation:** Multi-dimensional profiling across demographic cohorts, tenure maturity, balance tiers, and product usage.
2. **Empirical Balance Exposure Accounting:** Quantifying balance volume held by churned accounts (€185.59M portfolio total).
3. **Statistical Hypothesis Testing:** Rigorous Chi-square and Welch's t-tests identifying significant behavioral and demographic churn drivers.
4. **Interactive Executive BI Dashboard:** A 5-page responsive Streamlit application featuring dynamic KPI cards, interactive Plotly visualizations, cross-dimensional heatmaps, and high-value customer export capabilities.

---

## 🏛️ Project Architecture & Structure

```
customer-churn-european-banking/
│
├── data/
│   ├── raw/
│   │   └── European_Bank.csv              # Original 10,000-row bank dataset
│   └── processed/
│       └── European_Bank_cleaned.csv      # Cleaned & feature-engineered dataset
│
├── notebooks/
│   └── customer_churn_eda.ipynb           # Reproducible Jupyter EDA & Modeling
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                     # Safe path resolution & caching
│   ├── data_cleaning.py                   # Data validation, sanitation, privacy enforcement
│   ├── feature_engineering.py             # Dynamic percentile segmentation & high-value flags
│   ├── segmentation.py                    # 1D/2D aggregations & cross-tabulation engine
│   ├── churn_analysis.py                  # Statistical testing (t-test, Chi2) & ML baselines
│   ├── kpi_calculations.py                # Core portfolio, segment, and exposure KPI engine
│   ├── generate_charts.py                 # Static visualizer exporter
│   └── insights.py                        # Dynamic natural language narrative engine
│
├── dashboard/
│   ├── __init__.py
│   ├── charts.py                          # Reusable interactive Plotly charts & heatmaps
│   └── components.py                      # UI cards, modern banking styles, sidebar filters
│
├── reports/
│   ├── research_report.md                 # Complete academic & empirical research report
│   ├── executive_summary.md               # C-suite executive briefing document
│   └── key_insights.md                    # Data-driven summary of key findings
│
├── assets/
│   └── charts/                            # Exported high-resolution chart images
│
├── tests/
│   ├── test_data_cleaning.py              # Schema, null, and duplicate tests
│   ├── test_segmentation.py              # Segmentation thresholds & completeness tests
│   ├── test_kpis.py                       # KPI mathematical boundaries & safety tests
│   └── test_analysis.py                   # Hypothesis testing & ML baseline tests
│
├── app.py                                 # Main 5-Page Streamlit Analytics Application
├── requirements.txt                       # Project dependencies
├── README.md                              # Main documentation
├── .gitignore
└── LICENSE
```

---

## 📊 Dataset & Analytical Variables

The underlying dataset `European_Bank.csv` contains 10,000 rows and 14 attributes:

| Column | Type | Description |
| :--- | :--- | :--- |
| `Year` | Integer | Observation year (2025) |
| `CustomerId` | Integer | Unique customer identifier |
| `Surname` | String | Customer surname (*strictly excluded from analytical processing for privacy*) |
| `CreditScore` | Integer | Credit rating (350 to 850) |
| `Geography` | String | Country market (`France`, `Germany`, `Spain`) |
| `Gender` | String | Customer gender (`Male`, `Female`) |
| `Age` | Integer | Customer age (18 to 92) |
| `Tenure` | Integer | Years with the bank (0 to 10) |
| `Balance` | Float | Account deposit balance in Euros (€0 to €250,898.09) |
| `NumOfProducts`| Integer | Number of active bank products held (1 to 4) |
| `HasCrCard` | Binary | Credit card ownership flag (0 = No, 1 = Yes) |
| `IsActiveMember`| Binary | Active engagement indicator (0 = Inactive, 1 = Active) |
| `EstimatedSalary`| Float | Estimated annual salary in Euros |
| `Exited` | Binary | **Target Variable:** 0 = Retained customer, 1 = Churned customer |

---

## 🔬 Methodology & Dynamic Thresholds

All thresholds and financial segments are calculated dynamically from `European_Bank.csv` without artificial hardcoding:

### 1. Account Balance Segmentation
- **Zero Balance:** $\text{Balance} = 0$ (3,617 accounts, 36.17%)
- **Low Balance:** $0 < \text{Balance} \le €106,901.45$ (33rd non-zero percentile)
- **Medium Balance:** $€106,901.45 < \text{Balance} \le €131,564.57$ (66th non-zero percentile)
- **High Balance:** $\text{Balance} > €131,564.57$ (Top non-zero tertile)

### 2. Estimated Salary Segmentation
- **Low Salary:** $\text{Salary} \le €66,392.42$ (33rd percentile)
- **Medium Salary:** $€66,392.42 < \text{Salary} \le €131,508.52$ (66th percentile)
- **High Salary:** $\text{Salary} > €131,508.52$ (Top tertile)

### 3. High-Value Customer Definition
$$\text{IsHighValue} = (\text{Balance} > €131,564.57) \land (\text{EstimatedSalary} > €131,508.52)$$
- **Population:** 740 customers (7.40% of total portfolio)
- **Churn Count:** 180 accounts (**24.32% churn rate**)
- **High-Value Balance Exposure:** **€27,721,898.12**

### 4. Demographic & Tenure Segments
- **Age Group:** `<30` (7.56% churn), `30-45` (15.30% churn), `46-60` (**51.12% churn**), `60+` (24.78% churn).
- **Tenure Group:** `New (0-2)` (21.15% churn), `Mid-term (3-7)` (19.94% churn), `Long-term (8-10)` (20.45% churn).
- **Product Segment:** `1 Product` (27.71% churn), `2 Products` (**7.58% churn**), `3+ Products` (**85.89% churn** - small sample caution).

---

## 📈 Key Empirical Findings

1. **Overall Portfolio Health:** 20.37% overall churn rate (2,037 churners), representing €185.59M in balance exposure.
2. **Acute German Market Vulnerability:** German accounts exhibit an observed churn rate of **32.44%** (double France's 16.15% and Spain's 16.67%) with an average balance of €119,730.12.
3. **The 46–60 Age Risk Spike:** Over 1 in 2 customers aged 46–60 have churned (51.12%), identifying pre-retirement wealth transition as a critical attrition vector.
4. **The 2-Product Loyalty Anchor:** Customers with 2 products achieve a 7.58% churn rate, compared to 27.71% for single-product holders.
5. **Inactivity Penalty:** Inactive members churn at 26.85% (vs. 14.27% for active members), accounting for €118.53M in balance exposure.

---

## 🖥️ Streamlit Interactive Application

The interactive web dashboard includes 5 analytical modules:
1. **Executive Overview:** High-level KPI cards, Churn vs Retention Donut chart, country/age breakdown, and automated natural language insights.
2. **Customer Segmentation:** Deep-dive tabs for Age, Credit Score, Tenure, Balance, Salary, and Product Count with risk bubble charts and sortable metrics tables.
3. **Geography Analysis:** Multi-metric regional comparison across France, Germany, and Spain with cross-dimensional heatmaps (Geography $\times$ Age, Geography $\times$ Balance).
4. **Churn Drivers & Statistics:** Statistical profile comparisons, boxplot distributions, formal Chi-square and Welch's t-test hypothesis tables, and optional ML baseline benchmarks.
5. **High-Value Churn Explorer:** Balance exposure tracker, high-value customer registry (privacy-protected without Surnames), and CSV export functionality.

---

## 🚀 Quickstart & Installation

### Prerequisites
- Python 3.11+
- Git

### 1. Clone & Set Up Environment
```bash
git clone https://github.com/your-username/customer-churn-european-banking.git
cd customer-churn-european-banking

# Create and activate virtual environment (optional)
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```

### 3. Run Automated Tests
```bash
python -m pytest -v
```

---

## 🧪 Test Suite Coverage

The project includes an automated Pytest test suite covering:
- `test_data_cleaning.py`: Schema validation, 10,000-row assertion, 0 nulls, binary domain constraints, and privacy stripping.
- `test_segmentation.py`: Dynamic non-zero balance and salary quantile calculations, complete mutually exclusive categorizations.
- `test_kpis.py`: Mathematical consistency ($0 \le \text{rate} \le 100$, churn + retention = 100%), high-value metrics, zero-division error safety on empty filters.
- `test_analysis.py`: Aggregation sums, Chi-square & t-test execution, and ML model baseline pipelines.

---

## ⚖️ Statistical Disclaimer & Privacy Governance
- **Privacy:** Customer surnames are permanently removed from processed data and analytical views. Customer IDs are restricted to the High-Value Explorer.
- **Statistical Causality:** All findings represent observed statistical associations within historical observational data and do not imply direct causation.
- **Exposure Definition:** All financial exposure figures denote total account balance (€) held by churned customers and do not represent realized revenue loss.

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
