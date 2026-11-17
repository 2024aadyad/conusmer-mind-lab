import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Load dataset
df = pd.read_csv("data/raw/online_shoppers.csv")

print("STARTING CUSTOMER SEGMENTATION...")

# Select behavioural variables
features = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay"
]

X = df[features].copy()

# Standardise variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Customer_Segment"] = kmeans.fit_predict(X_scaled)

# Analyse the segments
print("\nCUSTOMER SEGMENTATION COMPLETE!")

print("\nCUSTOMER DISTRIBUTION:")

print(
    df["Customer_Segment"].value_counts().sort_index()
)

# Segment profiles
print("\nSEGMENT PROFILES:")

profile = df.groupby("Customer_Segment").agg(
    Total_Sessions=("Revenue", "count"),
    Purchase_Rate=("Revenue", "mean"),
    Average_Product_Pages=("ProductRelated", "mean"),
    Average_Product_Time=("ProductRelated_Duration", "mean"),
    Average_Bounce_Rate=("BounceRates", "mean"),
    Average_Page_Value=("PageValues", "mean")
)

profile["Purchase_Rate"] *= 100

print(profile.round(2))

print("\nMACHINE LEARNING SEGMENTATION SUCCESSFUL!")