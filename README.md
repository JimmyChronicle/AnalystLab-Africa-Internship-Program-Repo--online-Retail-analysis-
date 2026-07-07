# AnalystLab-Africa-Internship-Program-Repo--online-Retail-analysis-
Online Retail transactional data (541,909 rows)

# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 2. Load Dataset
df = pd.read_csv("OnlineRetail.csv", encoding="latin1")
print("Original Shape:", df.shape)

# 3. Data Cleaning
# Handle missing values
df_clean = df.dropna(subset=["CustomerID"]).copy()
df_clean["Description"] = df_clean["Description"].fillna("No Description")

# Remove duplicates
df_clean = df_clean.drop_duplicates()

# Data type conversion & standardization
df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"], errors="coerce")
df_clean["CustomerID"] = df_clean["CustomerID"].astype(int)
df_clean["Description"] = df_clean["Description"].str.strip().str.upper()
df_clean["Country"] = df_clean["Country"].str.strip().str.title()

# Flag cancellations
df_clean["IsCancellation"] = df_clean["InvoiceNo"].str.startswith("C", na=False) | (df_clean["Quantity"] < 0)

# Remove invalid prices
df_clean = df_clean[df_clean["UnitPrice"] > 0].copy()

print("Cleaned Shape:", df_clean.shape)

# Save cleaned data
df_clean.to_csv("OnlineRetail_cleaned.csv", index=False)

# 4. Feature Engineering
df_clean["TotalPrice"] = df_clean["Quantity"] * df_clean["UnitPrice"]
df_clean["YearMonth"] = df_clean["InvoiceDate"].dt.to_period("M")

# 5. Summary Statistics
print(df_clean[["Quantity", "UnitPrice", "TotalPrice"]].describe().round(2))

# 6. Exploratory Analysis & Visualizations
# 6.1 Top Products by Revenue
top_products = df_clean.groupby("Description")["TotalPrice"].sum().nlargest(10)

# 6.2 Monthly Sales Trend
monthly_sales = df_clean.groupby("YearMonth")["TotalPrice"].sum()

# 6.3 Revenue by Country
country_revenue = df_clean.groupby("Country")["TotalPrice"].sum().nlargest(5)

# 7. Visualizations
plt.style.use("default")
sns.set_style("whitegrid")

# Bar Chart
plt.figure(figsize=(12,6))
top_products.plot(kind="bar", color="skyblue")
plt.title("Top 10 Revenue-Generating Products")
plt.ylabel("Revenue (£)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("top_products_bar.png")
plt.close()

# Line Chart - Monthly Trend
plt.figure(figsize=(12,6))
monthly_sales.plot(kind="line", marker="o", color="darkblue")
plt.title("Monthly Revenue Trend (2010-2011)")
plt.ylabel("Total Revenue (£)")
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("monthly_sales_line.png")
plt.close()

# Pie Chart
plt.figure(figsize=(9,9))
country_revenue.plot(kind="pie", autopct="%1.1f%%")
plt.title("Revenue Share by Top 5 Countries")
plt.ylabel("")
plt.savefig("country_revenue_pie.png")
plt.close()

print("✅ Analysis Complete! Visualizations saved.")
