import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv(
    "data/raw/online_shoppers.csv"
)

print("=" * 50)
print("CONSUMER MIND LAB - DATA ANALYSIS")
print("=" * 50)

# 1. Dataset overview
print("\n1. DATASET OVERVIEW")

print("Total Sessions:", len(df))
print("Total Variables:", len(df.columns))

# 2. Purchase analysis
print("\n2. PURCHASE ANALYSIS")

purchase_count = df["Revenue"].sum()
total_sessions = len(df)

conversion_rate = (
    purchase_count / total_sessions
) * 100

print("Purchases:", purchase_count)
print("Non-Purchases:", total_sessions - purchase_count)

print("Conversion Rate:",
      round(conversion_rate, 2), "%")

# 3. Visitor analysis
print("\n3. VISITOR TYPE ANALYSIS")

visitor_analysis = df.groupby(
    "VisitorType"
)["Revenue"].mean() * 100

print(visitor_analysis)

# 4. Weekend vs weekday
print("\n4. WEEKEND VS WEEKDAY")

weekend_analysis = df.groupby(
    "Weekend"
)["Revenue"].mean() * 100

print(weekend_analysis)

# 5. Browsing behaviour
print("\n5. BROWSING BEHAVIOUR")

browsing_analysis = df.groupby(
    "Revenue"
)[[
    "Administrative_Duration",
    "Informational_Duration",
    "ProductRelated_Duration"
]].mean()

print(browsing_analysis)

# 6. Missing values
print("\n6. MISSING VALUES")

print(df.isnull().sum())

print("\nANALYSIS COMPLETED!")