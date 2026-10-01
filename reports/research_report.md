# Customer Segmentation & Churn Pattern Analytics in European Banking
## Comprehensive Research Report & Strategic Empirical Analysis

**Project Title:** Customer Segmentation & Churn Pattern Analytics in European Banking  
**Dataset:** `European_Bank.csv` (10,000 observations, 14 attributes)  
**Target Variable:** `Exited` (Binary: 0 = Retained, 1 = Churned)  
**Methodology:** Observational Data Analytics, Dynamic Feature Engineering, Non-parametric Risk Profiling, Statistical Hypothesis Testing  

---

## 1. Executive Summary

Customer attrition is one of the most substantial latent costs in retail banking. This research report presents an empirical evaluation of customer attrition dynamics across European retail banking operations in France, Germany, and Spain. Analyzing an audited cohort of 10,000 retail banking accounts, the study evaluates multi-dimensional churn risks across demographic cohorts, relationship duration, behavioral engagement, product portfolios, and financial wealth tiers.

### Key Empirical Findings:
- **Baseline Churn Rate:** The portfolio exhibits an overall churn rate of **20.37%** (2,037 churned customers out of 10,000), representing **€185,588,094.63** in total balance exposure held by churned accounts.
- **Geographic Vulnerability:** German accounts demonstrate an acute attrition rate of **32.44%** (814 churned / 2,509 accounts), double that of France (**16.15%**) and Spain (**16.67%**).
- **Demographic Vulnerability:** The **46–60 age cohort** represents the highest-risk demographic, with an observed churn rate of **51.12%** (842 churned / 1,647 accounts), compared to only **7.56%** among customers aged under 30.
- **Engagement Sensitivity:** Inactive accounts exhibit a **26.85%** churn rate, compared to **14.27%** among active members (a +12.58 percentage point risk differential).
- **High-Value Exposure:** Using data-driven upper financial tertiles (Balance $> €131,564.57$ and Estimated Salary $> €131,508.52$), 740 customers were categorized as high-value, among whom 180 churned (**24.32%** churn rate), generating **€27,721,898.12** in high-value balance exposure.

---

## 2. Business Context & Problem Statement

Retail banks incur high customer acquisition costs (CAC). Retaining established relationships is vital for deposit stability, loan origination pipelines, and cross-selling efficiency. When high-balance and mature customers exit, the bank loses not only customer count but substantial liquidity that supports its treasury balance sheet.

This analytics initiative addresses the following core business questions:
1. What is the baseline churn rate across the retail portfolio?
2. How does attrition vary across geographic territories?
3. Which age cohorts and demographic profiles exhibit elevated churn?
4. What is the empirical relationship between product holdings and retention?
5. How does account inactivity associate with relationship dissolution?
6. What is the total financial balance exposure associated with churned relationships?

---

## 3. Dataset Description & Data Quality Assessment

The analysis utilizes `European_Bank.csv`, containing exactly 10,000 records and 14 attributes:

| Column Name | Data Type | Analytical Role | Domain / Range |
| :--- | :--- | :--- | :--- |
| `Year` | Integer | Temporal Context | 2025 |
| `CustomerId` | Integer | Account Identifier | 15,565,701 – 15,815,690 (10,000 unique) |
| `Surname` | String | Customer Surname | Excluded from analytical calculations for privacy |
| `CreditScore` | Integer | Credit Rating | 350 – 850 (Mean: 650.5) |
| `Geography` | Categorical | Market Territory | France (5,014), Germany (2,509), Spain (2,477) |
| `Gender` | Categorical | Gender Identity | Male (5,457), Female (4,543) |
| `Age` | Integer | Customer Age | 18 – 92 (Mean: 38.9) |
| `Tenure` | Integer | Years with Bank | 0 – 10 (Mean: 5.0) |
| `Balance` | Float | Account Balance (€) | €0.00 – €250,898.09 (Mean: €76,485.89) |
| `NumOfProducts`| Integer | Products Held | 1 to 4 (Mean: 1.53) |
| `HasCrCard` | Binary | Credit Card Ownership | 0 (2,945), 1 (7,055) |
| `IsActiveMember`| Binary | Active Engagement | 0 (4,849), 1 (5,151) |
| `EstimatedSalary`| Float | Estimated Annual Salary | €11.58 – €199,992.48 (Mean: €100,090.24) |
| `Exited` | Binary | Retention Target | 0 = Retained (7,963), 1 = Churned (2,037) |

### Quality Audit:
- **Missing Values:** Exactly 0 missing values across all columns.
- **Duplicate Rows:** Exactly 0 duplicate records.
- **Integrity Checks:** CustomerId uniqueness verified (10,000 unique values). All binary indicators strictly adhere to $\{0, 1\}$.

---

## 4. Data Cleaning & Feature Engineering Methodology

### Data Cleaning:
1. Whitespace stripped across all categorical strings.
2. Explicit type casting enforced across integer, float, and categorical fields.
3. `Surname` dropped from analytical feature matrices to ensure data privacy governance.
4. `Year` preserved for provenance. Cleaned dataset saved to `data/processed/European_Bank_cleaned.csv`.

### Empirical Segmentation Thresholds:
All financial thresholds are computed dynamically from `European_Bank.csv`:

1. **Age Group:**
   - `<30`: Under 30 years (1,641 accounts, 16.41%)
   - `30-45`: 30 to 45 years (6,248 accounts, 62.48%)
   - `46-60`: 46 to 60 years (1,647 accounts, 16.47%)
   - `60+`: Over 60 years (464 accounts, 4.64%)

2. **Credit Score Band:**
   - `Low`: $< 580$ (2,362 accounts, 23.62%)
   - `Medium`: $580 \le \text{CreditScore} < 670$ (3,331 accounts, 33.31%)
   - `High`: $\ge 670$ (4,307 accounts, 43.07%)

3. **Tenure Group:**
   - `New (0-2)`: $\le 2$ years (2,496 accounts, 24.96%)
   - `Mid-term (3-7)`: $3 - 7$ years (5,005 accounts, 50.05%)
   - `Long-term (8-10)`: $\ge 8$ years (2,499 accounts, 24.99%)

4. **Account Balance Segmentation (Non-Zero Empirical Tertiles):**
   - Non-zero balance 33rd percentile ($P_{33}$): **€106,901.45**
   - Non-zero balance 66th percentile ($P_{66}$): **€131,564.57**
   - `Zero Balance`: $\text{Balance} = 0$ (3,617 accounts, 36.17%)
   - `Low Balance`: $0 < \text{Balance} \le €106,901.45$ (2,107 accounts, 21.07%)
   - `Medium Balance`: $€106,901.45 < \text{Balance} \le €131,564.57$ (2,106 accounts, 21.06%)
   - `High Balance`: $\text{Balance} > €131,564.57$ (2,170 accounts, 21.70%)

5. **Estimated Salary Segmentation (Dataset Tertiles):**
   - Salary 33rd percentile ($P_{33}$): **€66,392.42**
   - Salary 66th percentile ($P_{66}$): **€131,508.52**
   - `Low Salary`: $\le €66,392.42$ (3,300 accounts, 33.00%)
   - `Medium Salary`: $€66,392.42 < \text{Salary} \le €131,508.52$ (3,300 accounts, 33.00%)
   - `High Salary`: $\text{Salary} > €131,508.52$ (3,400 accounts, 34.00%)

6. **Product Segment:**
   - `1 Product`: 5,084 accounts (50.84%)
   - `2 Products`: 4,590 accounts (45.90%)
   - `3+ Products`: 326 accounts (3.26% — 266 with 3 products, 60 with 4 products)

7. **High-Value Customer Definition:**
   - **Formula:** $\text{Balance} > €131,564.57 \text{ (Top non-zero balance tier)} \land \text{EstimatedSalary} > €131,508.52 \text{ (Top salary tier)}$
   - **Population:** 740 customers (7.40% of total portfolio).

---

## 5. Exploratory Data Analysis & Empirical Findings

### 5.1 Overall Portfolio KPIs
- **Total Portfolio Size:** 10,000 customers
- **Total Portfolio Balance:** €764,858,892.88
- **Retained Customers:** 7,963 (79.63%)
- **Churned Customers:** 2,037 (20.37%)
- **Total Balance Exposure:** **€185,588,094.63**

### 5.2 Geographic Churn Analysis
| Geography | Customers | Churned | Churn Rate | Retained | Avg Balance | Avg CreditScore | Active Member Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Germany** | 2,509 | 814 | **32.44%** | 1,695 | €119,730.12 | 651.5 | 49.74% |
| **Spain** | 2,477 | 413 | **16.67%** | 2,064 | €61,818.14 | 651.3 | 52.97% |
| **France** | 5,014 | 810 | **16.15%** | 4,204 | €62,092.37 | 649.7 | 51.68% |

*Insight:* Germany's churn rate is exactly double that of France. German customers also maintain the highest average balance (€119,730.12 vs €62,092.37 in France), compounding financial balance exposure in the German market.

### 5.3 Demographic Analysis (Age & Gender)
| Demographic Segment | Customer Count | Churned Count | Churn Rate (%) | Population Share (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Age <30** | 1,641 | 124 | 7.56% | 16.41% |
| **Age 30–45** | 6,248 | 956 | 15.30% | 62.48% |
| **Age 46–60** | 1,647 | 842 | **51.12%** | 16.47% |
| **Age 60+** | 464 | 115 | 24.78% | 4.64% |
| **Female** | 4,543 | 1,139 | **25.07%** | 45.43% |
| **Male** | 5,457 | 898 | 16.46% | 54.57% |

*Insight:* The 46–60 cohort experiences more than 1 in 2 customers exiting (51.12%). Female customers exhibit an 8.61 percentage point higher churn rate than male customers.

### 5.4 Behavioral Engagement (Active vs. Inactive Members)
| Membership Status | Customers | Churned | Churn Rate (%) | Active Rate | Balance Exposure |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Inactive (0)** | 4,849 | 1,302 | **26.85%** | 0.00% | €118,525,834.12 |
| **Active (1)** | 5,151 | 735 | **14.27%** | 100.00% | €67,062,260.51 |

*Insight:* Inactive accounts exhibit nearly double the churn rate of active accounts, representing an empirical gap of +12.58 percentage points.

### 5.5 Product Portfolio Dynamics
| Product Segment | Customers | Churned | Churn Rate (%) | Retained | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1 Product** | 5,084 | 1,409 | 27.71% | 3,675 | Sub-optimal bundling |
| **2 Products** | 4,590 | 348 | **7.58%** | 4,242 | **Optimal Retention Anchor** |
| **3+ Products** | 326 | 280 | **85.89%** | 46 | Extreme Risk (Small sample) |

*Caution on 3+ Products:* While 3+ product holdings show an 85.89% churn rate, this group comprises 326 accounts (3.26% of portfolio). Customers with 2 products exhibit the lowest churn rate (7.58%).

### 5.6 High-Value Customer Risk & Balance Exposure
- **High-Value Customer Count:** 740
- **High-Value Churned Count:** 180
- **High-Value Churn Rate:** **24.32%** (vs. 20.37% portfolio average)
- **High-Value Share of Churners:** 8.84%
- **High-Value Balance Exposure:** **€27,721,898.12**

---

## 6. Statistical Hypothesis Testing

| Feature / Dimension | Test Executed | Test Statistic | p-Value | Significant ($\alpha=0.05$) | Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Geography** | Chi-Square Test | $\chi^2 = 301.255$ | $< 0.001$ | Yes | Strong geographic divergence in churn |
| **Gender** | Chi-Square Test | $\chi^2 = 112.919$ | $< 0.001$ | Yes | Statistically significant gender variance |
| **Age Cohort** | Chi-Square Test | $\chi^2 = 1045.022$| $< 0.001$ | Yes | Age is the single strongest categorical driver |
| **Active Membership**| Chi-Square Test | $\chi^2 = 242.986$ | $< 0.001$ | Yes | Inactivity strongly associates with exit |
| **Product Tier** | Chi-Square Test | $\chi^2 = 1503.666$| $< 0.001$ | Yes | Product count heavily associates with churn |
| **Balance Segment** | Chi-Square Test | $\chi^2 = 175.768$ | $< 0.001$ | Yes | Non-zero balances show higher exit rates |
| **Credit Card** | Chi-Square Test | $\chi^2 = 0.509$   | $0.475$ | **No** | Credit card ownership does not associate with churn |
| **Customer Age** | Welch's t-Test | $t = 29.77$       | $< 0.001$ | Yes | Churned mean age (44.8) > Retained (37.4) |
| **Account Balance** | Welch's t-Test | $t = 11.45$       | $< 0.001$ | Yes | Churned mean balance (€91.1k) > Retained (€72.7k) |
| **Credit Score** | Welch's t-Test | $t = -2.71$      | $0.007$ | Yes | Minor difference (Churned: 645.4 vs Retained: 651.9) |

> **Methodological Disclaimer:** Observed statistical associations indicate empirical relationships within historical data and do not imply direct causation.

---

## 7. Strategic Business Recommendations

Based on empirical evidence, six actionable retention strategies are recommended:

1. **Targeted Mature Customer Retention (Age 46–60):**
   - *Evidence:* 51.12% churn rate in the 46–60 age group.
   - *Action:* Deploy dedicated wealth-transition and pre-retirement banking packages with preferential term deposit rates and relationship managers.

2. **German Market Intervention:**
   - *Evidence:* 32.44% churn rate and €119.7k average balance in Germany.
   - *Action:* Review regional fee structures, competitor deposit offerings, and local digital banking UX in Germany.

3. **Systematic Inactive Account Re-Engagement:**
   - *Evidence:* Inactive accounts account for €118.5M in balance exposure with a 26.85% churn rate.
   - *Action:* Trigger automated re-activation outreach (e.g., cashback on debit usage, savings goal notifications) after 60 days of inactivity.

4. **Product Portfolio Optimization (The 2-Product Sweetspot):**
   - *Evidence:* Customers with 2 products achieve the lowest churn rate (7.58%), while 1-product accounts churn at 27.71%.
   - *Action:* Prioritize cross-selling a second core product (e.g., mortgage or credit facility to current account holders). Investigate friction in 3+ product onboarding.

5. **High-Value Client Relationship Governance:**
   - *Evidence:* 180 high-value churners represent €27.72M in balance exposure.
   - *Action:* Establish proactive quarterly portfolio reviews and dedicated private banking contact lines for accounts exceeding €131.5k in balance.

---

## 8. Limitations & Analytical Governance

- **Cross-Sectional Observational Scope:** The dataset represents a snapshot; dynamic time-series event tracking is limited.
- **Balance Exposure vs. Revenue:** Reported financial exposure denotes balance volume (€), not net fee revenue or interest margin.
- **Unobserved External Factors:** Competitor promotions, macroeconomic interest rate shifts, and localized service outages are not captured in the dataset.
