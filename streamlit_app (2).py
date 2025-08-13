
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --- Page Setup ---
st.set_page_config(page_title="Elasticity Tool", layout="wide")
st.title("📊 Elasticity Dashboard")

# --- Load Data ---
df = pd.read_csv("Price Elasticity Scores 2.csv")
df["item_price_bucket"] = df["item_price_bucket"].str.strip()

bucket_order = ["<= £3", "£3-£10", "£10-£20", "£20-£50", "£50+"]
df["bucket_order"] = df["item_price_bucket"].map({v: i for i, v in enumerate(bucket_order)})

# =======================
# 🧠 Module 0: Executive Summary
# =======================
st.header("🧠 Executive Summary: Buyer Sensitivity and Price Elasticity Post-SWIFT")

st.markdown("""
The introduction of the **SWIFT buyer fee** (flat £0.72 + 4% of item price) significantly impacted final prices, especially in low-price categories.
This triggered **non-uniform buyer reactions** across price tiers, captured through **price elasticity** — a measure of how demand responds to price changes.
""")

st.subheader("📊 What Is Price Elasticity?")
st.markdown("""
- **Definition**: Measures % change in quantity for a % change in price  
- **Example**: Elasticity = -2 → A 10% price increase causes 20% demand drop  
- **Relevance**: Helps prioritize pricing strategy, test design, and fee alignment
""")

# 📉 Conceptual Elasticity Curves (updated — no green line)
st.subheader("📉 Conceptual Demand Curves by Elasticity Type")
price = np.linspace(1, 100, 500)
quantity = 25000 / price  # Unit elastic: P * Q = constant

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=[500, 500],
    y=[0, 100],
    mode="lines",
    name="Perfectly Inelastic",
    line=dict(color="red", width=3, dash="dot")
))
fig.add_trace(go.Scatter(
    x=quantity,
    y=price,
    mode="lines",
    name="Unit Elastic",
    line=dict(color="blue", width=3)
))
fig.update_layout(
    title="Theoretical Demand Curves: Elasticity Types",
    xaxis_title="Quantity Demanded",
    yaxis_title="Price",
    xaxis=dict(range=[0, 1000]),
    yaxis=dict(range=[0, 100]),
    template="plotly_white",
    legend_title="Elasticity Type",
    width=800,
    height=500
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("💷 SWIFT Fee Impact by Price Tranche")
fee_impact_data = {
    "Item Price": ["£1.96", "£6.50", "£35.00", "£75.00"],
    "SWIFT Fee": ["£0.80", "£0.98", "£2.12", "£3.72"],
    "New Price": ["£2.76", "£7.48", "£37.12", "£78.72"],
    "% Increase": ["+40.7%", "+15.1%", "+6.1%", "+5.0%"]
}
st.table(pd.DataFrame(fee_impact_data))
st.markdown("🟠 **Takeaway**: Low-price items face disproportionately high price increases, making them more elastic.")

st.subheader("🧪 Model Comparison for Elasticity Estimation")
model_comparison = pd.DataFrame({
    "Criteria": [
        "Explainability", "Handles seasonality", "Non-linear effects",
        "Interacts with other variables", "Forecasting future impact",
        "Elasticity over price ranges", "Business interpretability"
    ],
    "Log-Log Regression": [
        "✅ High (easy to interpret coefficients)", "⚠️ Needs dummy vars", "❌ No",
        "⚠️ Limited", "⚠️ Linear extrapolation only", "❌ One average value", "✅ Strong"
    ],
    "Time Series (SARIMAX / Prophet)": [
        "⚠️ Moderate (if exogenous vars used)", "✅ Built-in", "❌ No",
        "⚠️ Some (via regressors)", "✅ Excellent", "⚠️ If modeled correctly", "⚠️ Moderate"
    ],
    "Random Forest": [
        "❌ Low (black box, needs SHAP)", "⚠️ Needs explicit features", "✅ Yes",
        "✅ Strong", "✅ Good (with proper data)", "✅ Can give local elasticities", "⚠️ Needs explanation tools"
    ]
})
st.dataframe(model_comparison, use_container_width=True)

# =======================
# 📦 Module 1: Elasticity Curve Viewer
# =======================
st.header("1️⃣ Elasticity Curve by Category")
selected_cat = st.selectbox("Select meta_categ_name", sorted(df["meta_categ_name"].unique()))
filtered_df = df[df["meta_categ_name"] == selected_cat].copy()
filtered_df.sort_values("bucket_order", inplace=True)

st.subheader("Elasticity Scores Table")
st.dataframe(
    filtered_df[[
        "meta_categ_name", "item_price_bucket",
        "Linear_Regression", "Time_Series", "Random_Forest"
    ]].reset_index(drop=True),
    use_container_width=True
)

melted = filtered_df.melt(
    id_vars=["item_price_bucket", "bucket_order"],
    value_vars=["Linear_Regression", "Time_Series", "Random_Forest"],
    var_name="Model",
    value_name="Elasticity"
)
melted.sort_values("bucket_order", inplace=True)

fig = px.line(
    melted,
    x="item_price_bucket",
    y="Elasticity",
    color="Model",
    markers=True,
    title=f"Elasticity Curve for {selected_cat}"
)
fig.update_layout(
    xaxis=dict(
        type="category",
        categoryorder="array",
        categoryarray=bucket_order
    )
)
st.plotly_chart(fig, use_container_width=True)

# =======================
# 📈 Module 2: Price Change Simulator
# =======================
st.markdown("---")
st.header("2️⃣ Price Change Impact Simulator")

with st.form("simulator_form"):
    st.markdown("Enter price change and select inputs to estimate volume and GMV impact.")
    col1, col2 = st.columns(2)
    with col1:
        sim_cat = st.selectbox("meta_categ_name", sorted(df["meta_categ_name"].unique()), key="sim_cat")
    with col2:
        sim_tranche = st.selectbox("item_price_bucket", sorted(df["item_price_bucket"].unique()), key="sim_tranche")
    sim_model = st.selectbox(
        "Elasticity Model",
        ["Linear_Regression", "Time_Series", "Random_Forest"]
    )
    price_change_pct = st.number_input("Enter % Price Change", value=-10.0, step=1.0)
    submitted = st.form_submit_button("Simulate")

if submitted:
    row = df[(df["meta_categ_name"] == sim_cat) & (df["item_price_bucket"] == sim_tranche)]
    if row.empty:
        st.warning("No data available for the selected combination.")
    else:
        elasticity = row[sim_model].values[0]
        baseline_volume = row["Weekly_Avg_BI"].values[0]
        asp = row["ASP"].values[0]
        demand_change_pct = elasticity * (price_change_pct / 100)
        new_volume = baseline_volume * (1 + demand_change_pct)
        gmv_change_weekly = (new_volume - baseline_volume) * asp
        gmv_change_annual = gmv_change_weekly * 52
        st.success(f"📦 Baseline Weekly Orders: {baseline_volume:.0f}")
        st.info(f"📉 Forecasted Weekly Orders: {new_volume:.0f}")
        if gmv_change_weekly >= 0:
            st.success(f"💰 Weekly GMV increase: £{gmv_change_weekly:,.2f}")
            st.success(f"📈 Yearly GMV impact: £{gmv_change_annual:,.2f}")
        else:
            st.error(f"💸 Weekly GMV loss: £{abs(gmv_change_weekly):,.2f}")
            st.error(f"📉 Yearly GMV impact: £{abs(gmv_change_annual):,.2f}")
