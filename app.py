import itertools
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Incentive Plan Design & Modeling", layout="wide")

st.title("Incentive Plan Design & Modeling")
st.write("Upload client data, select incentive metrics, and identify plan designs that best align payouts with shareholder value creation.")

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
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

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

def payout_curve(score, threshold, target, maximum):
    if score < threshold:
        return 0.0
    if score < target:
        return 0.5 + ((score - threshold) / (target - threshold)) * 0.5
    if score < maximum:
        return 1.0 + ((score - target) / (maximum - target)) * 1.0
    return 2.0

def run_model(growth, metrics, value_col, scenarios, target_incentive, threshold, target, maximum):
    cols = metrics + [value_col]
    means = growth[cols].mean().values
    cov = growth[cols].cov().values

    sims = np.random.multivariate_normal(means, cov, size=scenarios)
    sim_df = pd.DataFrame(sims, columns=cols)

    results = []
    weight_sets = generate_weight_sets(metrics)

    for weight_set in weight_sets:
        performance_score = np.zeros(len(sim_df))

        for metric, weight in weight_set.items():
            performance_score += (1 + sim_df[metric]) * weight

        payout_multiple = np.array([
            payout_curve(x, threshold, target, maximum)
            for x in performance_score
        ])

        payouts = payout_multiple * target_incentive
        value_change = sim_df[value_col]

        corr = np.corrcoef(payouts, value_change)[0, 1]
        if np.isnan(corr):
            corr = 0

        avg_payout = payouts.mean()
        volatility = payouts.std()

        alignment_score = (
            max(corr, 0) * 70
            + max(0, 1 - abs((avg_payout / target_incentive) - 1)) * 20
            + max(0, 1 - (volatility / target_incentive)) * 10
        )

        results.append({
            "metric_mix": " / ".join([f"{int(w*100)}% {m}" for m, w in weight_set.items()]),
            "alignment_score": round(alignment_score, 1),
            "pay_value_correlation": round(corr, 2),
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

st.subheader("Raw Data Preview")
st.dataframe(df, use_container_width=True)

num_cols = numeric_columns(df)

if len(num_cols) < 2:
    st.error("Your file needs at least two numeric columns.")
else:
    st.subheader("Column Mapping")

    date_col = st.selectbox("Select date / period column", df.columns)
    value_col = st.selectbox("Select shareholder value measure", num_cols)

    metric_options = [c for c in num_cols if c != value_col]

    selected_metrics = st.multiselect(
        "Select incentive metrics to test",
        metric_options,
        default=metric_options[: min(5, len(metric_options))]
    )

    df_sorted = df.copy()
    df_sorted[date_col] = pd.to_datetime(df_sorted[date_col], errors="coerce")
    df_sorted = df_sorted.sort_values(date_col)

    model_cols = selected_metrics + [value_col]
    model_df = df_sorted[[date_col] + model_cols].copy()

    for c in model_cols:
        model_df[c] = pd.to_numeric(model_df[c], errors="coerce")

    model_df = model_df.dropna()
    growth = model_df[model_cols].pct_change().replace([np.inf, -np.inf], np.nan).dropna()

    st.subheader("Sorted Modeling Data")
    st.dataframe(model_df, use_container_width=True)

    if len(growth) < 3:
        st.error("Not enough historical observations after cleaning. Use at least 5 periods if possible.")
    else:
        st.subheader("Historical Metric Relationship to Shareholder Value")

        corr_table = (
            growth.corr()[[value_col]]
            .drop(index=value_col)
            .rename(columns={value_col: "correlation_to_value"})
            .sort_values("correlation_to_value", ascending=False)
        )

        st.dataframe(corr_table, use_container_width=True)

        fig_corr = px.bar(
            corr_table.reset_index(),
            x="correlation_to_value",
            y="index",
            orientation="h",
            title="Historical Correlation to Shareholder Value"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        if st.button("Run Incentive Plan Model"):
            if len(selected_metrics) == 0:
                st.error("Select at least one incentive metric.")
            else:
                with st.spinner("Running covariance-based incentive plan scenarios..."):
                    results = run_model(
                        growth,
                        selected_metrics,
                        value_col,
                        scenarios,
                        target_incentive,
                        threshold,
                        target,
                        maximum
                    )

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
