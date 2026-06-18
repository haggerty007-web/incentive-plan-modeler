import itertools
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Incentive Plan Design & Modeling", layout="wide")

st.title("Incentive Plan Design & Modeling")
st.write("Upload historical client data, test incentive plan designs, and identify metric combinations that best align pay with shareholder value creation.")

REQUIRED_COLUMNS = ["year", "revenue", "ebitda", "eps", "free_cash_flow", "roic", "tsr"]

def sample_data():
    return pd.DataFrame({
        "year": [2019, 2020, 2021, 2022, 2023, 2024],
        "revenue": [1000, 970, 1080, 1160, 1240, 1310],
        "ebitda": [180, 160, 210, 235, 250, 270],
        "eps": [2.40, 2.10, 2.80, 3.10, 3.35, 3.60],
        "free_cash_flow": [90, 75, 105, 118, 130, 142],
        "roic": [9.5, 8.2, 10.1, 11.0, 11.6, 12.2],
        "tsr": [0.08, -0.12, 0.24, 0.11, 0.16, 0.13],
    })

def read_file(file):
    if file is None:
        return sample_data()
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    return pd.read_excel(file)

def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df

def generate_weight_sets(metrics, step=25, max_metrics=3):
    weight_sets = []
    for r in range(1, min(max_metrics, len(metrics)) + 1):
        for combo in itertools.combinations(metrics, r):
            if r == 1:
                weight_sets.append({combo[0]: 1.0})
            else:
                for weights in itertools.product(range(step, 101, step), repeat=r):
                    if sum(weights) == 100:
                        weight_sets.append(dict(zip(combo, [w / 100 for w in weights])))
    return weight_sets

def payout_curve(performance, threshold, target, maximum):
    if performance < threshold:
        return 0
    if performance < target:
        return 0.5 + ((performance - threshold) / (target - threshold)) * 0.5
    if performance < maximum:
        return 1.0 + ((performance - target) / (maximum - target)) * 1.0
    return 2.0

def run_model(df, metrics, scenarios, target_incentive, threshold, target, maximum):
    historical_growth = df[metrics + ["tsr"]].pct_change().dropna()
    results = []

    weight_sets = generate_weight_sets(metrics)

    for weight_set in weight_sets:
        payouts = []
        tsrs = []

        for _ in range(scenarios):
            simulated = {}
            for metric in metrics:
                mean = historical_growth[metric].mean()
                std = historical_growth[metric].std()
                simulated[metric] = np.random.normal(mean, std)

            tsr_mean = historical_growth["tsr"].mean()
            tsr_std = historical_growth["tsr"].std()
            simulated_tsr = np.random.normal(tsr_mean, tsr_std)

            weighted_performance = sum((1 + simulated[m]) * w for m, w in weight_set.items())
            payout_percent = payout_curve(weighted_performance, threshold, target, maximum)

            payouts.append(payout_percent * target_incentive)
            tsrs.append(simulated_tsr)

        correlation = np.corrcoef(payouts, tsrs)[0, 1]
        if np.isnan(correlation):
            correlation = 0

        avg_payout = np.mean(payouts)
        volatility = np.std(payouts)

        alignment_score = (
            max(correlation, 0) * 60
            + min(avg_payout / target_incentive, 2) * 20
            + max(0, 1 - volatility / max(target_incentive, 1)) * 20
        )

        results.append({
            "metric_mix": " / ".join([f"{int(w*100)}% {m}" for m, w in weight_set.items()]),
            "alignment_score": round(alignment_score, 1),
            "pay_tsr_correlation": round(correlation, 2),
            "average_payout": round(avg_payout, 0),
            "payout_volatility": round(volatility, 0),
        })

    return pd.DataFrame(results).sort_values("alignment_score", ascending=False)

with st.sidebar:
    st.header("Model Inputs")
    uploaded_file = st.file_uploader("Upload client historical data", type=["csv", "xlsx"])
    scenarios = st.slider("Number of scenarios", 500, 10000, 3000, step=500)
    target_incentive = st.number_input("Target incentive opportunity ($)", value=1000000, step=50000)

    st.subheader("Performance Curve")
    threshold = st.number_input("Threshold performance", value=0.90)
    target = st.number_input("Target performance", value=1.00)
    maximum = st.number_input("Maximum performance", value=1.20)

df = clean_columns(read_file(uploaded_file))

st.subheader("Data Preview")
st.dataframe(df)

missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
if missing:
    st.warning("Missing expected columns: " + ", ".join(missing))
    st.write("For best results, use columns named: year, revenue, ebitda, eps, free_cash_flow, roic, tsr")
else:
    available_metrics = ["revenue", "ebitda", "eps", "free_cash_flow", "roic"]

    selected_metrics = st.multiselect(
        "Select incentive metrics to test",
        available_metrics,
        default=["revenue", "ebitda", "eps", "free_cash_flow", "roic"]
    )

    if st.button("Run Incentive Plan Model"):
        with st.spinner("Running thousands of incentive plan scenarios..."):
            results = run_model(
                df,
                selected_metrics,
                scenarios,
                target_incentive,
                threshold,
                target,
                maximum
            )

        st.subheader("Top Recommended Plan Designs")
        st.dataframe(results.head(20), use_container_width=True)

        best = results.iloc[0]

        st.success(f"Best design: {best['metric_mix']}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Alignment Score", best["alignment_score"])
        col2.metric("Pay / TSR Correlation", best["pay_tsr_correlation"])
        col3.metric("Average Payout", f"${best['average_payout']:,.0f}")

        st.subheader("Alignment Score by Plan Design")
        fig = px.bar(
            results.head(15),
            x="alignment_score",
            y="metric_mix",
            orientation="h",
            title="Top 15 Incentive Plan Designs"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Pay-for-Performance Risk View")
        fig2 = px.scatter(
            results,
            x="average_payout",
            y="pay_tsr_correlation",
            size="alignment_score",
            hover_name="metric_mix",
            title="Average Payout vs Pay/TSR Correlation"
        )
        st.plotly_chart(fig2, use_container_width=True)

        csv = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Results",
            data=csv,
            file_name="incentive_plan_modeling_results.csv",
            mime="text/csv"
        )
