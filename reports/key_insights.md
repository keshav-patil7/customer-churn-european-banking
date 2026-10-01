# Key Empirical Findings: European Banking Customer Churn Analytics

This document summarizes the core empirical findings calculated dynamically from the 10,000-customer European Banking dataset.

---

### 1. Overall Portfolio Metrics
- **Total Customer Accounts:** 10,000
- **Retained Customers:** 7,963 (79.63%)
- **Churned Customers:** 2,037 (20.37%)
- **Average Credit Score:** 650.5
- **Average Age:** 38.9 years
- **Average Account Balance:** €76,485.89
- **Average Estimated Salary:** €100,090.24
- **Active Member Percentage:** 51.51%
- **Credit Card Ownership Rate:** 70.55%
- **Total Portfolio Balance:** €764,858,892.88
- **Total Balance Exposure (All Churners):** €185,588,094.63

---

### 2. Geographic Market Breakdown
- **Germany:** 2,509 accounts | 814 churned | **32.44% Churn Rate** | €119,730.12 Avg Balance | €97.46M Balance Exposure
- **Spain:** 2,477 accounts | 413 churned | **16.67% Churn Rate** | €61,818.14 Avg Balance | €25.53M Balance Exposure
- **France:** 5,014 accounts | 810 churned | **16.15% Churn Rate** | €62,092.37 Avg Balance | €50.29M Balance Exposure

---

### 3. Age Cohort Risk Hierarchy
- **Age <30:** 1,641 accounts | 124 churned | **7.56% Churn Rate**
- **Age 30–45:** 6,248 accounts | 956 churned | **15.30% Churn Rate**
- **Age 46–60:** 1,647 accounts | 842 churned | **51.12% Churn Rate** (Highest Risk)
- **Age 60+:** 464 accounts | 115 churned | **24.78% Churn Rate**

---

### 4. Product Holding Dynamics
- **1 Product:** 5,084 accounts | 1,409 churned | **27.71% Churn Rate**
- **2 Products:** 4,590 accounts | 348 churned | **7.58% Churn Rate** (Optimal Retention)
- **3+ Products:** 326 accounts | 280 churned | **85.89% Churn Rate** *(Flagged: small sample size, 3.26% of accounts)*

---

### 5. High-Value Customer Risk
- **High-Value Definition:** Balance $> €131,564.57$ (66th non-zero percentile) AND Estimated Salary $> €131,508.52$ (66th salary percentile).
- **High-Value Population:** 740 customers (7.40% of portfolio)
- **High-Value Churn Count:** 180 accounts
- **High-Value Churn Rate:** **24.32%**
- **High-Value Share of Total Churners:** 8.84%
- **High-Value Balance Exposure:** **€27,721,898.12**
