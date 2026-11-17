import pandas as pd
import plotly.express as px

# Load dataset
df = pd.read_csv("data/raw/online_shoppers.csv")

# Convert Revenue to readable labels
df["Purchase Status"] = df["Revenue"].map({
    True: "Purchased",
    False: "Did Not Purchase"
})

# 1. Purchase distribution

purchase_counts = df["Purchase Status"].value_counts().reset_index()
purchase_counts.columns = ["Purchase Status", "Sessions"]

fig1 = px.pie(
    purchase_counts,
    names="Purchase Status",
    values="Sessions",
    title="Consumer Purchase Distribution",
    hole=0.5
)

fig1.show()

# 2. Visitor type analysis

visitor_data = df.groupby(
    "VisitorType"
)["Revenue"].mean().reset_index()

visitor_data["Purchase Rate"] = visitor_data["Revenue"] * 100

fig2 = px.bar(
    visitor_data,
    x="VisitorType",
    y="Purchase Rate",
    title="Purchase Rate by Visitor Type",
    text_auto=".2f"
)

fig2.show()

# 3. Browsing duration comparison

browsing_data = df.groupby(
    "Purchase Status"
)[[
    "Administrative_Duration",
    "Informational_Duration",
    "ProductRelated_Duration"
]].mean().reset_index()

browsing_long = browsing_data.melt(
    id_vars="Purchase Status",
    var_name="Browsing Category",
    value_name="Average Duration"
)

fig3 = px.bar(
    browsing_long,
    x="Browsing Category",
    y="Average Duration",
    color="Purchase Status",
    barmode="group",
    title="Browsing Behaviour: Buyers vs Non-Buyers"
)

fig3.show()

print("VISUAL ANALYTICS COMPLETED!")