

# Consumer Mind Lab 🧠

An interactive consumer behaviour and marketing intelligence dashboard built with **Python, Streamlit and machine learning**.

Consumer Mind Lab explores how online shopping session data can be turned into useful marketing insights. It brings together exploratory data analysis, customer-session segmentation, purchase-likelihood estimation and campaign planning tools in one interactive app.

> **Project note:** This is a student-built learning prototype using a historical online shopping dataset. It is not connected to a live store, customer account or personal browsing history. Predictions and campaign simulations are exploratory estimates, not guarantees of real-world results.

## Features

- **Purchase overview** — Explore shopping sessions, purchases and the overall conversion rate.
- **Visitor insights** — Compare browsing and purchase patterns across visitor types.
- **Browsing activity** — Explore how visitors interact with different types of website pages.
- **Weekend shopping** — Compare shopping activity and purchase patterns on weekends and weekdays.
- **Customer-session segmentation** — Use K-Means clustering to group similar shopping sessions based on browsing behaviour. The groups represent sessions in the dataset, not verified individual customers.
- **Purchase predictor** — Use a trained Random Forest model to estimate the likelihood that a described shopping session results in a purchase.
- **Marketing strategy generator** — Explore rule-based marketing recommendations and build a campaign plan.
- **Campaign budget planner** — Experiment with hypothetical budget allocation across marketing channels.
- **Campaign performance simulator** — Estimate possible impressions, clicks, conversions and returns using assumptions entered by the user.
- **Strategy report** — Download a summary of the generated marketing plan.

## Built with

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Plotly
- Joblib

## Dataset

The project uses the **Online Shoppers Purchasing Intention** dataset from the UCI Machine Learning Repository. It contains 12,330 website sessions and includes browsing behaviour, visitor information and a `Revenue` target indicating whether a session resulted in a purchase.

The dataset is historical and is used for educational analysis. The app does not track visitors or collect live customer data.

## Run locally

### 1. Clone the repository

```bash
git clone https://github.com/2024aadyad/conusmer-mind-lab.git
cd consumer_mind_lab
```

### 2. Create and activate a virtual environment

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (usually `http://localhost:8501`).

## Project structure

```text
consumer_mind_lab/
├── app.py
├── data/
│   ├── raw/
│   │   └── online_shoppers.csv
│   └── processed/
├── models/
│   └── purchase_model.pkl
├── modules/
│   ├── campaign_analysis.py
│   ├── consumer_segmentation.py
│   ├── marketing_simulator.py
│   ├── pricing_psychology.py
│   └── purchase_prediction.py
├── requirements.txt
└── README.md
```

## Important limitations

- Purchase likelihoods are model estimates based on historical sample data; they are not certain outcomes.
- Customer segmentation groups website sessions, not known or identifiable people.
- Marketing recommendations are generated using transparent rules and should be treated as hypotheses to test.
- Budget allocations and campaign performance results are hypothetical scenarios based on user-entered assumptions, not actual campaign data.
- Model performance on a held-out test set does not guarantee performance on new or live business data.

## Purpose

This project was built to explore the intersection of **consumer behaviour, marketing analytics and machine learning**, and to practise turning data insights into practical marketing decisions.
