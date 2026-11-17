import pandas as pd
import os

# Your dataset location
file_path = "/Users/aadyadas/Downloads/online_shoppers_intention.csv"

print("Loading Consumer Behaviour Dataset...")

if os.path.exists(file_path):

    df = pd.read_csv(file_path)

    print("\nDATASET LOADED SUCCESSFULLY!")
    print("\nDataset Shape:", df.shape)

    print("\nFirst 5 Records:")
    print(df.head())

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nPurchase Distribution:")
    print(df["Revenue"].value_counts())

else:
    print("Dataset not found. Check the file path.")