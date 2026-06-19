import itertools
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Incentive Plan Design & Modeling", layout="wide")

st.title("Incentive Plan Design & Modeling")
st.write("Upload client data, select any incentive metrics, and test which combinations best align pay with shareholder value creation.")

def read_file(file):
    if file is None:
        return pd.DataFrame({
            "period": ["2020", "2021", "2022", "2023", "2024"],
            "revenue": [1000, 1080, 1160, 1240, 1310],
            "ebitda": [180, 210, 235, 250, 270],
            "eps": [2.40, 2.80, 3.10, 3.35, 3.60],
            "market_cap": [5000, 6200, 5900, 7100, 8200],
            "stock_price": [20, 25, 23, 29, 34],
        })
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
        .str.replace(".", "_")
    )
    return df

def numeric_columns(df):
    cols = []
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols

def generate_weight_sets(metrics, step=25, max_metrics=4):
    sets = []
    for r in range(1, min(max_metrics, len(metrics)) + 1):
        for combo in itertools.combinations(metrics, r):
            if r == 1:
                sets.append({combo[0]: 1.0})
            else:
                for weights in itertools.product(range(step, 101, step), repeat=r):
                    if sum(weights) == 100:
                        sets.append(dict(zip(combo, [w / 100 for w in weights])))
    return sets

def payout_curve(performance, threshold, target, maximum):
    if performance < threshold:
        return 0
    if performance < target:
        return 0.5 + ((performance - threshold) / (target - threshold)) * 0.5
    if performance < maximum:
        return 1.0 + ((performance - target) / (maximum - target)) * 1.0
    return 2.0

def run_model(df, metrics, value_col, scenarios, target_incentive, threshold, target, maximum):
    model_df = df[metrics + [value_col]].copy()
    model_df = model_df.apply(pd.to_numeric, errors="coerce").dropna()

    growth = model_df.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    if len(growth) < 2:
        st.error("Not enough numeric historical data to run the model. Try using at least 4-5 periods of data.")
        return pd.DataFrame()

    results = []
    weight_sets = generate_weight_sets(metrics)

    for weight_set in weight_sets:
        payouts = []
        value_changes = []

        for _ in range(scenarios):
            simulated = {}

            for metric in metrics:
                mean = growth[metric].mean()
                std = growth[metric].std()
                if pd.isna(std) or std == 0:
                    std = 0.01
                simulated[metric] = np.random.normal(mean, std)

            value_mean = growth[value_col].mean()
            value_std = growth[value_col].std()
            if pd.isna(value_std) or value_std == 0:
                value_std = 0.01

            simulated_value = np.random.normal(value_mean, value_std)

            weighted_performance = sum((1 + simulated[m]) * w for m, w in weight_set.items())
            payout_percent = payout_curve(weighted_performance, threshold, target, maximum)

            payouts.append(payout_percent * target_incentive)
            value_changes.append(simulated_value)

        correlation = np.corrcoef(payouts, value_changes)[0, 1]
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
            "pay_value_correlation": round(correlation, 2),
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
st.dataframe(df, use_container_width=True)

num_cols = numeric_columns(df)

if len(num_cols) < 2:
    st.error("Your file needs at least two numeric columns.")
else:
    st.subheader("Column Mapping")

    value_col = st.selectbox(
        "Select shareholder value / outcome measure",
        num_cols,
        index=0
    )

    metric_options = [c for c in num_cols if c != value_col]

    selected_metrics = st.multiselect(
        "Select incentive metrics to test",
        metric_options,
        default=metric_options[: min(5, len(metric_options))]
    )

    st.caption("Examples: revenue, EBITDA, EPS, free cash flow, ROIC, margin, market cap, stock price, relative TSR, or any numeric metric in your file.")

    if st.button("Run Incentive Plan Model"):
        if len(selected_metrics) == 0:
            st.error("Select at least one incentive metric.")
        else:
            with st.spinner("Running incentive plan scenarios..."):
                results = run_model(
                    df,
                    selected_metrics,
                    value_col,
                    scenarios,
                    target_incentive,
                    threshold,
                    target,
                    maximum
                )

            if not results.empty:
                st.subheader("Top Recommended Plan Designs")
                st.dataframe(results.head(25), use_container_width=True)

                best = results.iloc[0]
                st.success(f"Best design: {best['metric_mix']}")

                col1, col2, col3 = st.columns(3)
                col1.metric("Alignment Score", best["alignment_score"])
                col2.metric("Pay / Value Correlation", best["pay_value_correlation"])
                col3.metric("Average Payout", f"${best['average_payout']:,.0f}")

                st.subheader("Top Plan Designs")
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
                    y="pay_value_correlation",
                    size="alignment_score",
                    hover_name="metric_mix",
                    title="Average Payout vs Pay / Shareholder Value Correlation"
                )
                st.plotly_chart(fig2, use_container_width=True)

                csv = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download Results",
                    data=csv,
                    file_name="incentive_plan_modeling_results.csv",
                    mime="text/csv"
                )
