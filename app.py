import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from sklearn.preprocessing import StandardScaler

from sklearn.cluster import KMeans

st.set_page_config(
    page_title="Consumer Mind Lab",
    page_icon="🧠",
    layout="wide"
)

# CONSUMER MIND LAB — visual theme
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3, h4 {
        font-family: 'Space Grotesk', sans-serif !important;
        letter-spacing: -0.035em;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(20, 184, 166, 0.12), transparent 34%),
            radial-gradient(ellipse at 92% 8%, rgba(99, 102, 241, 0.10), transparent 30%),
            #0b1120;
        color: #e7edf7;
    }
    [data-testid="stHeader"] {
        background: rgba(11, 17, 32, 0.82);
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111c31 0%, #0d1526 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.15);
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #f8fafc;
    }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(23, 37, 62, 0.95), rgba(15, 27, 47, 0.95));
        border: 1px solid rgba(45, 212, 191, 0.20);
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 10px 28px rgba(0, 0, 0, 0.18);
    }
    [data-testid="stMetricLabel"] {
        color: #9fb0c8 !important;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-family: 'Space Grotesk', sans-serif;
    }
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(90deg, #0f9f91, #6366f1);
        color: white;
        border: 0;
        border-radius: 10px;
        font-weight: 700;
        min-height: 44px;
        transition: transform 160ms ease, filter 160ms ease;
    }
    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        filter: brightness(1.12);
        transform: translateY(-1px);
        color: white;
        border: 0;
    }
    [data-baseweb="select"] > div,
    [data-baseweb="input"] > div,
    [data-baseweb="textarea"] > div {
        background: #121f34;
        border-color: rgba(148, 163, 184, 0.25);
        border-radius: 10px;
    }
    [data-testid="stTextArea"] textarea {
        background: #121f34;
        border-radius: 10px;
    }
    [data-testid="stPlotlyChart"],
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 14px;
        overflow: hidden;
    }
    hr {
        border-color: rgba(148, 163, 184, 0.18);
    }
    .hero-panel {
        padding: 30px 32px;
        margin: 8px 0 24px 0;
        border: 1px solid rgba(45, 212, 191, 0.28);
        border-radius: 20px;
        background: linear-gradient(115deg, rgba(15, 159, 145, 0.17), rgba(99, 102, 241, 0.15) 70%, rgba(17, 28, 49, 0.8));
        box-shadow: 0 18px 45px rgba(0, 0, 0, 0.20);
    }
    .hero-kicker {
        color: #5eead4;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 8px;
    }
    .hero-title {
        color: #f8fafc;
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(2rem, 4vw, 3rem);
        font-weight: 700;
        line-height: 1.05;
        letter-spacing: -0.05em;
        margin: 0 0 10px 0;
    }
    .hero-copy {
        color: #b7c5d9;
        font-size: 1rem;
        margin: 0;
    }
    .nav-label {
        color: #5eead4;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.15em;
        margin: 22px 0 10px 0;
    }
    .sidebar-note {
        color: #91a3bd;
        font-size: 0.78rem;
        line-height: 1.55;
        padding: 14px 12px;
        margin-top: 18px;
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 12px;
        background: rgba(15, 27, 47, 0.55);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] > div {
        gap: 7px;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label {
        background: rgba(30, 45, 68, 0.55);
        border: 1px solid rgba(148, 163, 184, 0.12);
        border-radius: 10px;
        padding: 9px 11px;
        transition: background 150ms ease, border-color 150ms ease;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
        background: rgba(20, 184, 166, 0.12);
        border-color: rgba(45, 212, 191, 0.35);
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label p {
        color: #dce7f5;
        font-weight: 600;
        font-size: 0.88rem;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(100deg, rgba(15, 159, 145, 0.25), rgba(99, 102, 241, 0.20));
        border-color: rgba(45, 212, 191, 0.55);
    }
    .section-intro {
        color: #aebed2;
        font-size: 0.98rem;
        line-height: 1.6;
        margin-top: -4px;
        margin-bottom: 22px;
    }
    .eyebrow {
        color: #5eead4;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 5px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load data
df = pd.read_csv("data/raw/online_shoppers.csv")
purchase_model = joblib.load(
    "models/purchase_model.pkl"
)
# CUSTOMER SEGMENTATION MODEL

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

X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

df["Customer_Segment"] = kmeans.fit_predict(X_scaled)

# Header
st.markdown(
    """
    <div class="hero-panel">
        <div class="hero-kicker">Consumer intelligence platform</div>
        <div class="hero-title">CONSUMER MIND LAB</div>
        <p class="hero-copy">Decode shopping behaviour. Predict purchase intent. Build smarter marketing campaigns.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ======== Quick-start guide for new users ========

with st.expander("👋 New here? Start with this quick guide", expanded=True):
    guide_col1, guide_col2, guide_col3 = st.columns(3)
    with guide_col1:
        st.markdown("**1. Explore the data**")
        st.write("See what the shopping sessions in our sample dataset tell us about visitors and purchases.")
    with guide_col2:
        st.markdown("**2. Try a prediction**")
        st.write("Change a few details about a sample website visit and see the model's estimated chance of purchase.")
    with guide_col3:
        st.markdown("**3. Build a campaign**")
        st.write("Turn the insights into marketing ideas, a sample budget and a hypothetical campaign forecast.")

st.info(
    "This is a learning prototype using a historical sample dataset. It does not connect to a live store or read anyone's personal browsing history. "
    "The predictions and campaign figures are estimates for exploration, not guaranteed results."
)

# Key metrics
total = len(df)
purchases = int(df["Revenue"].sum())
conversion = purchases / total * 100

c1, c2, c3 = st.columns(3)

c1.metric("Shopping Sessions", f"{total:,}")
c2.metric("Successful Purchases", f"{purchases:,}")
c3.metric("Conversion Rate", f"{conversion:.2f}%")

st.divider()

# Sidebar navigation
st.sidebar.markdown("## 🧠 Consumer Mind Lab")
st.sidebar.caption("MARKETING INTELLIGENCE SUITE")
st.sidebar.markdown("<div class='nav-label'>EXPLORE</div>", unsafe_allow_html=True)

module_options = [
    "Purchase Distribution",
    "Visitor Behaviour",
    "Browsing Behaviour",
    "Weekend Shopping",
    "Customer Segmentation",
    "Purchase Predictor",
    "Marketing Strategy Generator",
]

module_labels = {
    "Purchase Distribution": "🛍️  Purchase overview",
    "Visitor Behaviour": "👥  Visitor insights",
    "Browsing Behaviour": "🔎  Browsing activity",
    "Weekend Shopping": "📅  Weekend shopping",
    "Customer Segmentation": "🧩  Customer groups",
    "Purchase Predictor": "🎯  Purchase predictor",
    "Marketing Strategy Generator": "📣  Marketing planner",
}

choice = st.sidebar.radio(
    "Choose a section",
    module_options,
    format_func=lambda item: module_labels[item],
    label_visibility="collapsed",
)
st.sidebar.markdown(
    "<div class='sidebar-note'>Choose a section to explore shopping patterns, predict purchase interest, or plan a campaign.</div>",
    unsafe_allow_html=True,
)

if choice == "Purchase Distribution":

    st.header("🛍️ Purchase overview")
    st.markdown("<p class='section-intro'>See how many shopping visits ended in a purchase and how many did not.</p>", unsafe_allow_html=True)

    counts = df["Revenue"].value_counts().reset_index()
    counts.columns = ["Purchase Status", "Sessions"]

    fig = px.pie(
        counts,
        names="Purchase Status",
        values="Sessions",
        hole=0.55
    )

    st.plotly_chart(fig, use_container_width=True)

elif choice == "Visitor Behaviour":

    st.header("👥 Visitor insights")
    st.markdown("<p class='section-intro'>Compare how often first-time and returning visitors make a purchase.</p>", unsafe_allow_html=True)

    visitor = df.groupby(
        "VisitorType"
    )["Revenue"].mean().reset_index()

    visitor["Purchase Rate"] = visitor["Revenue"] * 100

    fig = px.bar(
        visitor,
        x="VisitorType",
        y="Purchase Rate",
        text_auto=".2f",
        title="Which visitors are more likely to purchase?"
    )

    st.plotly_chart(fig, use_container_width=True)

elif choice == "Browsing Behaviour":

    st.header("🔎 Browsing activity")
    st.markdown("<p class='section-intro'>Compare the time buyers and non-buyers spend exploring different parts of the website.</p>", unsafe_allow_html=True)

    browsing = df.groupby("Revenue")[
        [
            "Administrative_Duration",
            "Informational_Duration",
            "ProductRelated_Duration"
        ]
    ].mean().reset_index()

    browsing["Purchase Status"] = browsing["Revenue"].map({
        True: "Purchased",
        False: "Did Not Purchase"
    })

    long_data = browsing.melt(
        id_vars="Purchase Status",
        var_name="Category",
        value_name="Duration"
    )

    fig = px.bar(
        long_data,
        x="Category",
        y="Duration",
        color="Purchase Status",
        barmode="group"
    )

    st.plotly_chart(fig, use_container_width=True)

elif choice == "Weekend Shopping":

    st.header("📅 Weekend shopping")
    st.markdown("<p class='section-intro'>Find out whether visitors are more likely to buy on weekends or weekdays.</p>", unsafe_allow_html=True)

    weekend = df.groupby("Weekend")[
        "Revenue"
    ].mean().reset_index()

    weekend["Purchase Rate"] = weekend["Revenue"] * 100

    weekend["Day"] = weekend["Weekend"].map({
        True: "Weekend",
        False: "Weekday"
    })

    fig = px.bar(
        weekend,
        x="Day",
        y="Purchase Rate",
        text_auto=".2f"
    )

    st.plotly_chart(fig, use_container_width=True)
elif choice == "Customer Segmentation":

    st.header("🧩 Customer groups")

    st.markdown(
        "<p class='section-intro'>Group similar shopping visits together to understand different browsing and buying patterns. The app uses a machine-learning method called K-Means to create four groups.</p>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Each row represents a website visit, not a verified individual customer. "
        "The four groups are patterns found by the model in the sample data."
    )

    # Calculate segment profiles
    segment_profile = df.groupby("Customer_Segment").agg(
        Total_Sessions=("Revenue", "count"),
        Purchase_Rate=("Revenue", "mean"),
        Product_Pages=("ProductRelated", "mean"),
        Product_Time=("ProductRelated_Duration", "mean"),
        Bounce_Rate=("BounceRates", "mean"),
        Page_Value=("PageValues", "mean")
    ).reset_index()

    segment_profile["Purchase_Rate"] *= 100

    # Segment distribution
    st.subheader("1. How visits are split across groups")

    fig1 = px.pie(
        segment_profile,
        names="Customer_Segment",
        values="Total_Sessions",
        hole=0.55,
        title="Distribution of Shopping Sessions",
        color="Customer_Segment"
    )

    st.plotly_chart(fig1, use_container_width=True)

    # Purchase rate comparison
    st.subheader("2. Purchase rate in each group")

    fig2 = px.bar(
        segment_profile,
        x="Customer_Segment",
        y="Purchase_Rate",
        text_auto=".2f",
        title="Which Segment Converts Best?",
        labels={
            "Customer_Segment": "Customer Segment",
            "Purchase_Rate": "Purchase Rate (%)"
        }
    )

    st.plotly_chart(fig2, use_container_width=True)

    # Interactive segment explorer
    st.subheader("3. Explore a group")

    selected_segment = st.selectbox(
        "Choose a group to explore",
        sorted(df["Customer_Segment"].unique())
    )

    selected_data = segment_profile[
        segment_profile["Customer_Segment"] == selected_segment
    ].iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Shopping Sessions",
        f"{int(selected_data['Total_Sessions']):,}"
    )

    col2.metric(
        "Purchase Rate",
        f"{selected_data['Purchase_Rate']:.2f}%"
    )

    col3.metric(
        "Average Page Value",
        f"{selected_data['Page_Value']:.2f}"
    )

    st.subheader("4. Group details")

    st.dataframe(
        segment_profile.round(3),
        use_container_width=True,
        hide_index=True
    )

    st.success(
        f"You are viewing group {selected_segment}."
    )

elif choice == "Purchase Predictor":

    st.header("🎯 Purchase predictor")
    st.markdown(
        "<p class='section-intro'>Change the details below to estimate how likely this shopping visit is to end in a purchase. The result is a model estimate, not a promise.</p>",
        unsafe_allow_html=True,
    )

    st.subheader("Tell us about the shopping visit")
    st.caption(
        "You do not need real customer information to try this demo. Use the example values below, "
        "or change them to explore how different website visits affect the model's estimate."
    )

    with st.form("purchase_prediction_form"):
        left, right = st.columns(2)

        with left:
            administrative = st.number_input(
                "Administrative pages visited",
                min_value=0,
                max_value=int(df["Administrative"].max()),
                value=int(df["Administrative"].median()),
                step=1,
            )
            administrative_duration = st.number_input(
                "Time on administrative pages (seconds)",
                min_value=0.0,
                max_value=float(df["Administrative_Duration"].max()),
                value=float(df["Administrative_Duration"].median()),
                step=10.0,
            )
            informational = st.number_input(
                "Informational pages visited",
                min_value=0,
                max_value=int(df["Informational"].max()),
                value=int(df["Informational"].median()),
                step=1,
            )
            informational_duration = st.number_input(
                "Time on informational pages (seconds)",
                min_value=0.0,
                max_value=float(df["Informational_Duration"].max()),
                value=float(df["Informational_Duration"].median()),
                step=10.0,
            )
            product_related = st.number_input(
                "Product-related pages visited",
                min_value=0,
                max_value=int(df["ProductRelated"].max()),
                value=int(df["ProductRelated"].median()),
                step=1,
            )
            product_related_duration = st.number_input(
                "Time on product pages (seconds)",
                min_value=0.0,
                max_value=float(df["ProductRelated_Duration"].max()),
                value=float(df["ProductRelated_Duration"].median()),
                step=30.0,
            )

        with right:
            bounce_rate = st.number_input(
                "Bounce rate",
                min_value=float(df["BounceRates"].min()),
                max_value=float(df["BounceRates"].max()),
                value=float(df["BounceRates"].median()),
                step=0.001,
                format="%.4f",
            )
            exit_rate = st.number_input(
                "Exit rate",
                min_value=float(df["ExitRates"].min()),
                max_value=float(df["ExitRates"].max()),
                value=float(df["ExitRates"].median()),
                step=0.001,
                format="%.4f",
            )
            page_value = st.number_input(
                "Page value",
                min_value=float(df["PageValues"].min()),
                max_value=float(df["PageValues"].max()),
                value=float(df["PageValues"].median()),
                step=1.0,
            )
            special_day = st.number_input(
                "Special-day proximity (0 to 1)",
                min_value=0.0,
                max_value=1.0,
                value=float(df["SpecialDay"].median()),
                step=0.1,
            )
            visitor_type = st.selectbox(
                "Visitor type",
                options=sorted(df["VisitorType"].dropna().unique().tolist()),
            )
            weekend = st.selectbox(
                "Shopping session on a weekend?",
                options=[False, True],
                format_func=lambda value: "Yes" if value else "No",
            )

        submitted = st.form_submit_button(
            "Estimate purchase likelihood",
            use_container_width=True,
        )

    if submitted:
        session_input = pd.DataFrame([{
            "Administrative": administrative,
            "Administrative_Duration": administrative_duration,
            "Informational": informational,
            "Informational_Duration": informational_duration,
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_related_duration,
            "BounceRates": bounce_rate,
            "ExitRates": exit_rate,
            "PageValues": page_value,
            "SpecialDay": special_day,
            "VisitorType": visitor_type,
            "Weekend": weekend,
        }])

        purchase_probability = float(
            purchase_model.predict_proba(session_input)[0, 1]
        )
        # Keep the latest model result available when navigating to other modules.
        st.session_state["latest_purchase_probability"] = purchase_probability
        st.session_state["latest_session_input"] = session_input.copy()

        st.divider()
        st.subheader("Your result")
        result_col, gauge_col = st.columns([1, 2])

        with result_col:
            st.metric(
                "Estimated chance of purchase",
                f"{purchase_probability:.1%}",
            )
            if purchase_probability >= 0.5:
                st.success("The model classifies this session as more likely to purchase.")
            else:
                st.info("The model classifies this session as less likely to purchase.")

        with gauge_col:
            gauge = px.bar(
                x=[purchase_probability * 100],
                y=["Purchase probability"],
                orientation="h",
                range_x=[0, 100],
                text=[f"{purchase_probability:.1%}"],
                labels={"x": "Estimated probability (%)", "y": ""},
                title="Estimated likelihood of purchase",
            )
            gauge.update_traces(textposition="inside")
            gauge.update_layout(showlegend=False, height=180)
            st.plotly_chart(gauge, use_container_width=True)

        st.caption(
            "This prediction is generated by a Random Forest model trained on "
            "historical shopping sessions. It is an exploratory estimate, "
            "not a guarantee or a causal explanation."
        )

elif choice == "Marketing Strategy Generator":

    st.header("📣 Marketing planner")
    st.markdown(
        "<p class='section-intro'>Turn shopping patterns into practical campaign ideas. Choose an audience, review the shopping signals, and build a campaign plan step by step.</p>",
        unsafe_allow_html=True,
    )

    st.subheader("Choose your audience and shopping details")
    st.caption(
        "For this prototype, these are example scenario settings—not data collected from a live store. "
        "The app uses them to demonstrate how a marketer could turn customer insights into a campaign."
    )

    col1, col2 = st.columns(2)

    with col1:
        audience_type = st.selectbox(
            "Who are you trying to reach?",
            ["New visitors", "Returning visitors", "All visitors"],
        )
        if "latest_purchase_probability" in st.session_state:
            purchase_likelihood = round(
                st.session_state["latest_purchase_probability"] * 100
            )
            st.metric(
                "Latest AI-predicted purchase likelihood",
                f"{purchase_likelihood}%",
            )
            st.caption(
                "Automatically taken from your most recent Purchase Predictor result. "
                "Run a new prediction there to update this value."
            )
        else:
            purchase_likelihood = st.slider(
                "Example purchase likelihood (%) — use this only if you have not run the predictor",
                min_value=0,
                max_value=100,
                value=30,
            )
            st.info(
                "No AI prediction is available in this session yet. "
                "This temporary demo value is manual, not a model prediction. "
                "Visit Purchase Predictor and submit a session to connect the AI result."
            )
        bounce_percentage = st.slider(
            "How often do visitors leave without exploring further? (%)",
            min_value=0,
            max_value=100,
            value=35,
        )

    with col2:
        product_browsing_minutes = st.slider(
            "Average time spent looking at products (minutes)",
            min_value=0,
            max_value=60,
            value=5,
        )
        page_value_score = st.slider(
            "Website page value score",
            min_value=0.0,
            max_value=float(max(10, df["PageValues"].max())),
            value=float(min(2.0, max(0.0, df["PageValues"].max()))),
            step=0.5,
        )

    generate_clicked = st.button(
        "Build my marketing plan",
        use_container_width=True,
    )

    if generate_clicked:
        st.session_state["show_campaign_results"] = True

    if st.session_state.get("show_campaign_results", False):
        recommendations = []
        rationale = []

        if purchase_likelihood >= 70:
            recommendations.append((
                "High purchase intent",
                "Test relevant product bundles, complementary add-ons and "
                "clear checkout calls to action. Avoid unnecessary discounts "
                "when visitors already show strong intent."
            ))
            rationale.append(
                f"The model estimates a {purchase_likelihood}% purchase likelihood, "
                "which is in the high-intent range (70% or above). The strategy "
                "therefore focuses on helping interested visitors complete a purchase "
                "rather than relying on broad discounts."
            )
        elif purchase_likelihood >= 35:
            recommendations.append((
                "Moderate purchase intent",
                "Use personalised product recommendations, social proof and "
                "helpful product comparisons to reduce hesitation."
            ))
            rationale.append(
                f"The model estimates a {purchase_likelihood}% purchase likelihood, "
                "which falls in the moderate range (35% to 69%). Social proof and "
                "comparisons are suggested to help reduce possible hesitation."
            )
        else:
            recommendations.append((
                "Low purchase intent",
                "Test educational content, clearer value propositions and "
                "low-commitment offers to help visitors move further through "
                "the buying journey."
            ))
            rationale.append(
                f"The model estimates a {purchase_likelihood}% purchase likelihood, "
                "below 35%. The suggested first step is to build awareness and "
                "communicate value before asking visitors to purchase."
            )

        if bounce_percentage >= 50:
            recommendations.append((
                "High bounce rate",
                "Review landing-page relevance, page speed, mobile usability "
                "and whether the ad promise matches the page content."
            ))
            rationale.append(
                f"The selected bounce rate is {bounce_percentage}%, which meets "
                "the high-bounce threshold of 50%. This makes landing-page relevance, "
                "speed and usability useful areas to investigate."
            )
        elif bounce_percentage <= 20:
            recommendations.append((
                "Relatively low bounce rate",
                "Visitors appear to be exploring beyond the landing page. "
                "Test clearer navigation and relevant next-step prompts."
            ))
            rationale.append(
                f"The selected bounce rate is {bounce_percentage}%, at or below "
                "the 20% low-bounce threshold. The suggestion focuses on helping "
                "visitors take a useful next step."
            )

        if product_browsing_minutes >= 10:
            recommendations.append((
                "Extended product research",
                "Test retargeting with the products viewed, comparison guides, "
                "reviews and answers to common purchase objections."
            ))
            rationale.append(
                f"Visitors are set to spend {product_browsing_minutes} minutes on "
                "product pages, meeting the 10-minute extended-research threshold. "
                "Helpful reviews and comparisons may address questions during research."
            )
        elif product_browsing_minutes <= 2:
            recommendations.append((
                "Limited product-page engagement",
                "Improve product discovery with stronger category navigation, "
                "clearer product images and concise benefit-led descriptions."
            ))
            rationale.append(
                f"The selected product-page time is {product_browsing_minutes} minutes, "
                "at or below the 2-minute threshold. The recommendation focuses on "
                "making relevant products easier to discover and understand."
            )

        if page_value_score >= 5:
            recommendations.append((
                "Higher-value page activity",
                "Prioritise relevant cross-sells and a smooth checkout journey "
                "while monitoring average order value and conversion."
            ))
            rationale.append(
                f"The selected page value score is {page_value_score:.1f}, meeting "
                "the threshold of 5. The suggested actions focus on supporting "
                "relevant add-ons and a smooth checkout."
            )
        else:
            recommendations.append((
                "Opportunity to improve page value",
                "Review which pages contribute to valuable sessions and test "
                "more relevant product links and calls to action."
            ))
            rationale.append(
                f"The selected page value score is {page_value_score:.1f}, below "
                "5. The recommendation suggests reviewing which pages contribute "
                "to valuable sessions and improving relevant navigation."
            )

        if audience_type == "New visitors":
            recommendations.append((
                "New-visitor strategy",
                "Build trust quickly with clear delivery and return information, "
                "reviews and a simple explanation of the brand's value."
            ))
            rationale.append(
                "The selected audience is new visitors. Since they may not know "
                "the brand yet, the strategy prioritises trust signals and clear "
                "information about the brand and its policies."
            )
        elif audience_type == "Returning visitors":
            recommendations.append((
                "Returning-visitor strategy",
                "Test personalised reminders, recently viewed products and "
                "relevant loyalty benefits without over-messaging visitors."
            ))
            rationale.append(
                "The selected audience is returning visitors. The strategy uses "
                "relevant reminders and recently viewed products to reconnect "
                "visitors with products they have already explored."
            )
        else:
            recommendations.append((
                "Audience strategy",
                "Compare new and returning visitors separately before choosing "
                "one campaign message for the whole audience."
            ))
            rationale.append(
                "The selected audience includes all visitors. New and returning "
                "visitors can behave differently, so comparing them separately "
                "can help avoid using one message for everyone."
            )

        st.divider()
        st.subheader("Your marketing suggestions")

        report_lines = [
            "CONSUMER MIND LAB — MARKETING STRATEGY REPORT",
            f"Audience type: {audience_type}",
            f"Estimated purchase likelihood: {purchase_likelihood}%",
            f"Bounce rate: {bounce_percentage}%",
            f"Product browsing time: {product_browsing_minutes} minutes",
            f"Page value score: {page_value_score:.1f}",
            "",
            "RECOMMENDATIONS",
        ]

        for number, ((heading, action), reason) in enumerate(
            zip(recommendations, rationale), start=1
        ):
            st.markdown(f"**{number}. {heading}**")
            st.write(action)
            st.caption(f"Why this may help: {reason}")
            report_lines.extend([
                f"{number}. {heading}",
                action,
                f"Why this may help: {reason}",
                "",
            ])

        # Build a practical campaign from the same scenario inputs.
        if purchase_likelihood < 35:
            campaign_objective = "Build awareness and move visitors into consideration"
            campaign_channel = "Instagram / short-form video and educational landing-page content"
            campaign_message = (
                "Discover what makes the product useful. Explore the details, "
                "see how it fits your needs and make an informed choice."
            )
            campaign_cta = "Explore the collection"
            campaign_kpis = "Reach, engagement rate, product-page visits and add-to-cart rate"
        elif purchase_likelihood < 70:
            campaign_objective = "Nurture interest and reduce purchase hesitation"
            campaign_channel = "Email and consent-based retargeting with product-focused social proof"
            campaign_message = (
                "Still considering your options? See product reviews, compare "
                "key features and find the option that suits you best."
            )
            campaign_cta = "Compare products"
            campaign_kpis = "Click-through rate, return visits, add-to-cart rate and conversion rate"
        else:
            campaign_objective = "Convert high-intent visitors efficiently"
            campaign_channel = "On-site product recommendations and consent-based cart or browse reminders"
            campaign_message = (
                "Found something you like? Take another look at your selected "
                "product and complete your order when you are ready."
            )
            campaign_cta = "Continue to checkout"
            campaign_kpis = "Checkout completion, conversion rate and average order value"

        if audience_type == "New visitors":
            campaign_message += " New here? Find clear delivery, returns and product information before you decide."
        elif audience_type == "Returning visitors":
            campaign_message += " Welcome back — revisit products you have already explored."

        channel_options = [
            "Instagram / short-form video",
            "Email marketing",
            "Google Ads / paid search",
            "LinkedIn",
            "Website / on-site recommendations",
            "WhatsApp (with marketing consent)",
        ]

        if purchase_likelihood < 35:
            default_channel = "Instagram / short-form video"
        elif purchase_likelihood < 70:
            default_channel = "Email marketing"
        else:
            default_channel = "Website / on-site recommendations"

        campaign_channel = st.selectbox(
            "Choose your campaign channel",
            options=channel_options,
            index=channel_options.index(default_channel),
        )

        if bounce_percentage >= 50:
            campaign_channel += " — review landing-page relevance, speed and mobile usability first"
            campaign_kpis += "; landing-page engagement and bounce rate"

        if product_browsing_minutes >= 10:
            campaign_message += " Use detailed comparisons and customer reviews to support your research."
        elif product_browsing_minutes <= 2:
            campaign_message += " Make the main product benefits easy to understand at a glance."

        if page_value_score < 5:
            campaign_objective += "; improve discovery of relevant, valuable pages"

        campaign_duration = "14-day test, followed by a performance review"

        st.divider()
        st.subheader("🚀 Campaign Builder")
        st.write(
            "A campaign draft based on the audience and shopping details you selected. You can edit the message and choose a channel."
        )

        campaign_col1, campaign_col2 = st.columns(2)
        with campaign_col1:
            st.markdown("**What this campaign aims to do**")
            st.write(campaign_objective)
            st.markdown("**Who the campaign is for**")
            st.write(f"{audience_type} with an estimated {purchase_likelihood}% purchase likelihood")
            st.markdown("**Where to run it**")
            st.write(campaign_channel)
            st.markdown("**How long to test it**")
            st.write(campaign_duration)

        with campaign_col2:
            st.markdown("**Write your campaign message**")
            campaign_message = st.text_area(
                "Edit the message people will see",
                value=campaign_message,
                height=150,
                key="campaign_message_editor",
            )
            st.markdown("**Main button or action**")
            st.write(campaign_cta)
            st.markdown("**What to measure**")
            st.write(campaign_kpis)

        st.divider()
        st.subheader("💰 Campaign Budget Planner")
        st.write(
            "Set a total budget, then move the sliders to decide how much goes to each channel. The app will calculate the amounts for you."
        )

        total_budget = st.number_input(
            "Total amount to spend (₹)",
            min_value=1000,
            max_value=10000000,
            value=50000,
            step=5000,
            key="campaign_total_budget",
        )

        budget_col1, budget_col2 = st.columns(2)
        with budget_col1:
            instagram_weight = st.slider(
                "Instagram — budget share",
                min_value=0,
                max_value=100,
                value=50,
                key="budget_instagram_weight",
            )
            google_weight = st.slider(
                "Google Ads — budget share",
                min_value=0,
                max_value=100,
                value=25,
                key="budget_google_weight",
            )
            email_weight = st.slider(
                "Email — budget share",
                min_value=0,
                max_value=100,
                value=15,
                key="budget_email_weight",
            )
            other_weight = st.slider(
                "Other tests — budget share",
                min_value=0,
                max_value=100,
                value=10,
                key="budget_other_weight",
            )

        weight_total = (
            instagram_weight + google_weight + email_weight + other_weight
        )

        if weight_total == 0:
            st.warning(
                "All allocation weights are zero. Increase at least one weight "
                "to distribute the budget."
            )
            budget_allocation = pd.DataFrame(
                columns=["Channel", "Allocation (%)", "Budget (₹)"]
            )
        else:
            budget_weights = {
                "Instagram / short-form video": instagram_weight,
                "Google Ads": google_weight,
                "Email marketing": email_weight,
                "Other / testing": other_weight,
            }
            budget_allocation = pd.DataFrame([
                {
                    "Channel": channel,
                    "Allocation (%)": weight / weight_total * 100,
                    "Budget (₹)": total_budget * weight / weight_total,
                }
                for channel, weight in budget_weights.items()
                if weight > 0
            ])

            budget_allocation["Allocation (%)"] = budget_allocation[
                "Allocation (%)"
            ].round(1)
            budget_allocation["Budget (₹)"] = budget_allocation[
                "Budget (₹)"
            ].round(0).astype(int)

            budget_display_col, budget_chart_col = st.columns([1, 1])
            with budget_display_col:
                st.dataframe(
                    budget_allocation,
                    use_container_width=True,
                    hide_index=True,
                )
            with budget_chart_col:
                budget_chart = px.pie(
                    budget_allocation,
                    names="Channel",
                    values="Budget (₹)",
                    hole=0.5,
                    title="How your budget is split",
                )
                st.plotly_chart(budget_chart, use_container_width=True)

            st.caption(
                "This is an illustrative starting allocation, not a forecast of "
                "channel performance. Review actual costs and campaign results "
                "before committing spend."
            )

        report_lines.extend([
            "",
            "CAMPAIGN BUILDER",
            f"Campaign objective: {campaign_objective}",
            f"Target audience: {audience_type} ({purchase_likelihood}% estimated purchase likelihood)",
            f"Recommended channel: {campaign_channel}",
            f"Suggested campaign message: {campaign_message}",
            f"Call to action: {campaign_cta}",
            f"KPIs to track: {campaign_kpis}",
            f"Campaign duration: {campaign_duration}",
            "",
            "CAMPAIGN BUDGET PLANNER",
            f"Total campaign budget: ₹{total_budget:,.0f}",
        ])

        if not budget_allocation.empty:
            for _, budget_row in budget_allocation.iterrows():
                report_lines.append(
                    f"{budget_row['Channel']}: {budget_row['Allocation (%)']:.1f}% "
                    f"(₹{budget_row['Budget (₹)']:,.0f})"
                )
        else:
            report_lines.append("No budget allocated because all channel weights are zero.")

        report_lines.append("")

        st.divider()
        st.subheader("📊 Campaign Performance Simulator")
        st.write(
            "Try different assumptions to see how a campaign might perform. These are rough estimates for learning and planning, not guaranteed results."
        )
        st.info(
            "These numbers are calculated from the assumptions you choose below. "
            "They are not actual campaign results or forecasts from the purchase model."
        )

        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            assumed_cpm = st.number_input(
                "Estimated cost for 1,000 ad views (₹)",
                min_value=1.0,
                max_value=100000.0,
                value=150.0,
                step=10.0,
                key="sim_cpm",
            )
            assumed_ctr = st.slider(
                "Estimated percentage of viewers who click (%)",
                min_value=0.1,
                max_value=20.0,
                value=1.5,
                step=0.1,
                key="sim_ctr",
            )
            assumed_conversion_rate = st.slider(
                "Estimated percentage of clicks that lead to a purchase (%)",
                min_value=0.1,
                max_value=30.0,
                value=2.0,
                step=0.1,
                key="sim_conversion_rate",
            )
        with sim_col2:
            assumed_order_value = st.number_input(
                "Average amount spent per order (₹)",
                min_value=1.0,
                max_value=10000000.0,
                value=2500.0,
                step=100.0,
                key="sim_order_value",
            )
            assumed_frequency = st.number_input(
                "Average times each person sees the ad",
                min_value=1.0,
                max_value=20.0,
                value=2.0,
                step=0.5,
                key="sim_frequency",
            )

        estimated_impressions = total_budget / assumed_cpm * 1000
        estimated_reach = estimated_impressions / assumed_frequency
        estimated_clicks = estimated_impressions * (assumed_ctr / 100)
        estimated_conversions = estimated_clicks * (assumed_conversion_rate / 100)
        estimated_revenue = estimated_conversions * assumed_order_value
        estimated_roas = estimated_revenue / total_budget
        estimated_roi = (estimated_revenue - total_budget) / total_budget * 100

        st.markdown("**Estimated campaign outcomes**")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        metric_col1.metric("Estimated ad views", f"{estimated_impressions:,.0f}")
        metric_col2.metric("Estimated people reached", f"{estimated_reach:,.0f}")
        metric_col3.metric("Estimated ad clicks", f"{estimated_clicks:,.0f}")

        metric_col4, metric_col5, metric_col6 = st.columns(3)
        metric_col4.metric("Estimated purchases", f"{estimated_conversions:,.0f}")
        metric_col5.metric("Estimated sales value", f"₹{estimated_revenue:,.0f}")
        metric_col6.metric("Sales per ₹1 spent (ROAS)", f"{estimated_roas:.2f}x")

        st.metric("Estimated return after ad spend (ROI)", f"{estimated_roi:.1f}%")

        funnel_data = pd.DataFrame({
            "Stage": ["Impressions", "Clicks", "Conversions"],
            "Estimated volume": [
                estimated_impressions,
                estimated_clicks,
                estimated_conversions,
            ],
        })
        funnel_chart = px.bar(
            funnel_data,
            x="Stage",
            y="Estimated volume",
            text_auto=".0f",
            title="From ad views to purchases (estimate)",
        )
        st.plotly_chart(funnel_chart, use_container_width=True)

        st.caption(
            "Calculation assumptions: impressions = budget ÷ CPM × 1,000; "
            "reach = impressions ÷ frequency; clicks = impressions × CTR; "
            "conversions = clicks × conversion rate; revenue = conversions × average order value. "
            "ROI here compares estimated revenue with media budget only and excludes "
            "product costs, agency fees, taxes and other expenses. Replace assumptions "
            "with your own historical or platform data before making spending decisions."
        )

        report_lines.extend([
            "CAMPAIGN PERFORMANCE SIMULATOR (ILLUSTRATIVE)",
            f"Total budget: ₹{total_budget:,.0f}",
            f"Assumed CPM: ₹{assumed_cpm:,.2f}",
            f"Assumed CTR: {assumed_ctr:.1f}%",
            f"Assumed conversion rate: {assumed_conversion_rate:.1f}%",
            f"Assumed average order value: ₹{assumed_order_value:,.2f}",
            f"Assumed frequency: {assumed_frequency:.1f}",
            f"Estimated ad views: {estimated_impressions:,.0f}",
            f"Estimated people reached: {estimated_reach:,.0f}",
            f"Estimated ad clicks: {estimated_clicks:,.0f}",
            f"Estimated purchases: {estimated_conversions:,.0f}",
            f"Estimated sales value: ₹{estimated_revenue:,.0f}",
            f"Sales per ₹1 spent (ROAS): {estimated_roas:.2f}x",
            f"Estimated return after ad spend (ROI): {estimated_roi:.1f}%",
            "",
        ])

        report_lines.extend([
            "NOTE: These are rule-based exploratory recommendations, not "
            "proven causal effects or guaranteed campaign outcomes. Test "
            "recommendations with controlled experiments and real campaign data."
        ])

        st.download_button(
            "Download marketing strategy report",
            data="\n".join(report_lines),
            file_name="consumer_mind_lab_marketing_strategy.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.caption(
            "Recommendations are generated from the scenario inputs using "
            "transparent rules. They are strategic hypotheses to test, not "
            "predictions of guaranteed campaign performance."
        )

st.divider()

st.caption(
    "CONSUMER MIND LAB · A student-built learning prototype using Python, Streamlit and machine learning · Historical sample data, not live customer tracking"
)