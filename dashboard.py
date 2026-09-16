import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from rainfall_model import RainfallPredictor

# ============================================================
# INDIA RAINFALL ANALYSIS DASHBOARD
# ============================================================

st.set_page_config(
    page_title="India Rainfall Dashboard",
    page_icon="🌧️",
    layout="wide"
)

# ------------------------------------------------------------
# LOAD DATASETS
# ------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

ds1_path = PROJECT_DIR / "rainfall in india 1901-2015.csv"
ds2_path = PROJECT_DIR / "district_wise_rainfall_normal[1].csv"

df1 = pd.read_csv(ds1_path)
df2 = pd.read_csv(ds2_path)

# ------------------------------------------------------------
# TITLE
# ------------------------------------------------------------

st.title("🌧️ India Rainfall Prediction & Analysis Dashboard")
st.markdown(
    "Explore historical rainfall patterns across subdivisions, "
    "states and districts of India."
)

st.divider()

# ------------------------------------------------------------
# SUMMARY CARDS
# ------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Historical Records",
        f"{len(df1):,}"
    )

with col2:
    st.metric(
        "District Records",
        f"{len(df2):,}"
    )

with col3:
    st.metric(
        "Average Annual Rainfall",
        f"{df1['ANNUAL'].mean():,.2f} mm"
    )

with col4:
    st.metric(
        "Wettest Month",
        "July"
    )

st.divider()

# ------------------------------------------------------------
# STATE / DISTRICT SELECTION
# ------------------------------------------------------------

st.header("📍 District Profile")

states = sorted(df2["STATE_UT_NAME"].dropna().unique())

selected_state = st.selectbox(
    "Select State / Union Territory",
    states
)

state_data = df2[df2["STATE_UT_NAME"] == selected_state]

districts = sorted(state_data["DISTRICT"].dropna().unique())

selected_district = st.selectbox(
    "Select District",
    districts
)

district_data = state_data[
    state_data["DISTRICT"] == selected_district
].iloc[0]

st.subheader(
    f"{selected_district} — {selected_state}"
)

c1, c2, c3 = st.columns(3)

with c1:
    st.metric(
        "Annual Rainfall",
        f"{district_data['ANNUAL']:.1f} mm"
    )

with c2:
    st.metric(
        "Jun-Sep Rainfall",
        f"{district_data['Jun-Sep']:.1f} mm"
    )

with c3:
    st.metric(
        "Mar-May Rainfall",
        f"{district_data['Mar-May']:.1f} mm"
    )

# ------------------------------------------------------------
# MONTHLY RAINFALL FOR SELECTED DISTRICT
# ------------------------------------------------------------

months = [
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"
]

monthly_values = [
    district_data[m] for m in months
]

monthly_df = pd.DataFrame({
    "Month": months,
    "Rainfall": monthly_values
})

fig_month = px.bar(
    monthly_df,
    x="Month",
    y="Rainfall",
    title=f"Monthly Rainfall — {selected_district}",
    labels={
        "Rainfall": "Rainfall (mm)",
        "Month": "Month"
    }
)

# ───────── Seasonal subdivisions ─────────

# JAN–FEB: Winter
fig_month.add_vrect(
    x0=-0.5,
    x1=1.5,
    fillcolor="lightblue",
    opacity=0.12,
    line_width=1,
    line_color="lightblue"
)

# MAR–MAY: Pre-Monsoon
fig_month.add_vrect(
    x0=1.5,
    x1=4.5,
    fillcolor="lightgreen",
    opacity=0.12,
    line_width=1,
    line_color="lightgreen"
)

# JUN–SEP: Monsoon
fig_month.add_vrect(
    x0=4.5,
    x1=8.5,
    fillcolor="lightyellow",
    opacity=0.15,
    line_width=1,
    line_color="lightyellow"
)

# OCT–DEC: Post-Monsoon
fig_month.add_vrect(
    x0=8.5,
    x1=11.5,
    fillcolor="lightpink",
    opacity=0.12,
    line_width=1,
    line_color="lightpink"
)

# Seasonal labels
fig_month.add_annotation(
    x=0.5,
    y=1.08,
    xref="x",
    yref="paper",
    text="WINTER (JAN–FEB)",
    showarrow=False
)

fig_month.add_annotation(
    x=3,
    y=1.08,
    xref="x",
    yref="paper",
    text="PRE-MONSOON (MAR–MAY)",
    showarrow=False
)

fig_month.add_annotation(
    x=6.5,
    y=1.08,
    xref="x",
    yref="paper",
    text="MONSOON (JUN–SEP)",
    showarrow=False
)

fig_month.add_annotation(
    x=10,
    y=1.08,
    xref="x",
    yref="paper",
    text="POST-MONSOON (OCT–DEC)",
    showarrow=False
)

fig_month.update_layout(
    height=550,
    margin=dict(
        l=70,
        r=40,
        t=120,
        b=80
    ),
    xaxis=dict(
        title="Month",
        type="category",
        categoryorder="array",
        categoryarray=months,
        tickmode="array",
        tickvals=months,
        automargin=True
    ),
    yaxis=dict(
        title="Rainfall (mm)",
        automargin=True
    ),
    bargap=0.18
)

st.plotly_chart(
    fig_month,
    use_container_width=True,
    config={"responsive": True}
)

st.divider()
# ------------------------------------------------------------
# STATE RAINFALL COMPARISON
# ------------------------------------------------------------

st.header("🌦️ State / UT Rainfall Comparison")

state_summary = (
    df2.groupby("STATE_UT_NAME")["ANNUAL"]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

fig_state = px.bar(
    state_summary.head(15),
    x="STATE_UT_NAME",
    y="ANNUAL",
    title="Top 15 States / UTs by Average Annual Rainfall",
    labels={
        "STATE_UT_NAME": "State / UT",
        "ANNUAL": "Average Annual Rainfall (mm)"
    }
)

fig_state.update_layout(
    xaxis_tickangle=-45
)

st.plotly_chart(
    fig_state,
    use_container_width=True
)

st.divider()

# ------------------------------------------------------------
# HISTORICAL YEARLY TREND
# ------------------------------------------------------------

st.header("📈 Historical Rainfall Trend")

yearly_rainfall = (
    df1.groupby("YEAR")["ANNUAL"]
    .mean()
    .reset_index()
)

fig_year = px.line(
    yearly_rainfall,
    x="YEAR",
    y="ANNUAL",
    title="Average Annual Rainfall — 1901 to 2015",
    labels={
        "YEAR": "Year",
        "ANNUAL": "Average Rainfall (mm)"
    }
)

fig_year.update_traces(
    mode="lines"
)

st.plotly_chart(
    fig_year,
    use_container_width=True
)

st.divider()

# ------------------------------------------------------------
# INDIA MONTHLY RAINFALL PATTERN
# ------------------------------------------------------------

st.header("☔ India Monthly Rainfall Pattern")

national_monthly = (
    df1[months]
    .mean()
    .sort_values(ascending=False)
    .reset_index()
)

national_monthly.columns = [
    "Month",
    "Average Rainfall"
]

fig_national = px.bar(
    national_monthly,
    x="Month",
    y="Average Rainfall",
    title="Average Monthly Rainfall Across India",
    labels={
        "Average Rainfall": "Rainfall (mm)"
    }
)

st.plotly_chart(
    fig_national,
    use_container_width=True
)

st.divider()

# ------------------------------------------------------------
# TOP DISTRICTS
# ------------------------------------------------------------

st.header("🏆 Highest Rainfall Districts")

top_districts = (
    df2[
        ["STATE_UT_NAME", "DISTRICT", "ANNUAL"]
    ]
    .sort_values("ANNUAL", ascending=False)
    .head(10)
)

st.dataframe(
    top_districts,
    use_container_width=True,
    hide_index=True
)

# ------------------------------------------------------------
# RAINFALL PREDICTION
# ------------------------------------------------------------

st.divider()

st.header("🌧️ Rainfall Prediction")
st.markdown(
    "Predict annual rainfall for any Indian District using a Machine Learning "
    "(Random Forest) model trained on 115 years of meteorological records."
)

@st.cache_resource
def load_prediction_model():
    return RainfallPredictor(ds1_path, ds2_path)

predictor = load_prediction_model()

# User Inputs: State, District, Year
col_pred1, col_pred2, col_pred3 = st.columns(3)

with col_pred1:
    pred_states = sorted(df2["STATE_UT_NAME"].dropna().unique())
    pred_selected_state = st.selectbox(
        "Select State / UT for Prediction",
        pred_states,
        index=pred_states.index(selected_state) if selected_state in pred_states else 0,
        key="pred_state_select"
    )

with col_pred2:
    pred_state_districts = sorted(
        df2[df2["STATE_UT_NAME"] == pred_selected_state]["DISTRICT"].dropna().unique()
    )
    pred_default_idx = (
        pred_state_districts.index(selected_district)
        if selected_district in pred_state_districts else 0
    )
    pred_selected_district = st.selectbox(
        "Select District for Prediction",
        pred_state_districts,
        index=pred_default_idx,
        key="pred_district_select"
    )

with col_pred3:
    pred_selected_year = st.slider(
        "Select Year",
        min_value=1901,
        max_value=2035,
        value=2025,
        step=1,
        key="pred_year_slider"
    )

# Run Prediction
pred_result = predictor.predict(
    pred_selected_state,
    pred_selected_district,
    pred_selected_year
)

st.subheader(
    f"Prediction for {pred_selected_district} ({pred_selected_state}) — Year {pred_selected_year}"
)

# Display Key Prediction Metrics
kpi_c1, kpi_c2, kpi_c3, kpi_c4 = st.columns(4)

with kpi_c1:
    st.metric(
        "Predicted Annual Rainfall",
        f"{pred_result['predicted_annual']:,.1f} mm"
    )

with kpi_c2:
    st.metric(
        "District Normal Baseline",
        f"{pred_result['district_normal']:,.1f} mm"
    )

with kpi_c3:
    st.metric(
        "Expected Category",
        pred_result['category'],
        delta=f"{pred_result['pct_deviation']:+.1f}% vs Normal"
    )

with kpi_c4:
    if pred_result['actual_annual'] is not None:
        st.metric(
            f"Historical Actual ({pred_selected_year})",
            f"{pred_result['actual_annual']:,.1f} mm",
            delta=f"{pred_result['predicted_annual'] - pred_result['actual_annual']:+.1f} mm err"
        )
    else:
        st.metric(
            f"Projection Mode ({pred_selected_year})",
            "Future Forecast",
            delta="Iterative ML Lag"
        )

# Category Classification Banner
if pred_result['category'] == "Normal Rainfall":
    st.success(
        f"🟢 **{pred_result['category']}**: Expected rainfall is {pred_result['predicted_annual']:.1f} mm "
        f"({pred_result['pct_deviation']:+.1f}% deviation from normal {pred_result['district_normal']:.1f} mm). "
        f"This falls within the IMD standard Normal range (±19% of climatological normal)."
    )
elif pred_result['category'] == "Low Rainfall":
    st.warning(
        f"🟠 **{pred_result['category']} (Deficient)**: Expected rainfall is {pred_result['predicted_annual']:.1f} mm "
        f"({pred_result['pct_deviation']:+.1f}% deviation from normal {pred_result['district_normal']:.1f} mm). "
        f"This indicates significant deficit risk exceeding the IMD -19% threshold."
    )
else:
    st.info(
        f"🔵 **{pred_result['category']} (Excess)**: Expected rainfall is {pred_result['predicted_annual']:.1f} mm "
        f"({pred_result['pct_deviation']:+.1f}% deviation from normal {pred_result['district_normal']:.1f} mm). "
        f"This indicates surplus/heavy rainfall risk exceeding the IMD +19% threshold."
    )

# Model Performance Metrics
st.markdown("##### 📊 Model Performance Metrics")
m_c1, m_c2, m_c3, m_c4 = st.columns(4)

with m_c1:
    st.metric("Model Algorithm", "Random Forest Regressor")

with m_c2:
    st.metric("Mean Absolute Error (MAE)", f"{pred_result['metrics']['mae']} mm")

with m_c3:
    st.metric("Root Mean Squared Error (RMSE)", f"{pred_result['metrics']['rmse']} mm")

with m_c4:
    st.metric("R² Score (Variance Explained)", f"{pred_result['metrics']['r2']:.4f}")

# Predicted vs Actual Rainfall Chart
st.markdown("##### 📈 Predicted vs Actual Rainfall History & Forecast")

plot_data = pred_result["plot_df"].copy()
plot_data = plot_data.rename(columns={
    "ACTUAL_ANNUAL": "Actual Rainfall (mm)",
    "PREDICTED_ANNUAL": "ML Predicted Rainfall (mm)"
})

fig_pred_chart = px.line(
    plot_data,
    x="YEAR",
    y=["Actual Rainfall (mm)", "ML Predicted Rainfall (mm)"],
    title=f"Annual Rainfall: Actual vs ML Predicted — {pred_selected_district}, {pred_selected_state}",
    labels={"value": "Annual Rainfall (mm)", "YEAR": "Year", "variable": "Legend"},
    color_discrete_map={
        "Actual Rainfall (mm)": "#2E86C1",
        "ML Predicted Rainfall (mm)": "#E67E22"
    }
)

# Add normal baseline reference line
fig_pred_chart.add_hline(
    y=pred_result["district_normal"],
    line_dash="dash",
    line_color="#27AE60",
    annotation_text=f"District Normal ({pred_result['district_normal']:.1f} mm)",
    annotation_position="bottom right"
)

# Highlight selected year
fig_pred_chart.add_vline(
    x=pred_selected_year,
    line_dash="dot",
    line_color="#C0392B",
    annotation_text=f"Selected Year ({pred_selected_year})",
    annotation_position="top left"
)

fig_pred_chart.update_layout(
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    hovermode="x unified"
)

st.plotly_chart(
    fig_pred_chart,
    use_container_width=True
)
# ------------------------------------------------------------
# FUTURE RAINFALL FORECAST
# ------------------------------------------------------------

st.divider()

st.header("📅 Future Rainfall Forecast")

st.markdown(
    f"""
    Generate a Machine Learning based rainfall projection for
    **{pred_selected_district}, {pred_selected_state}** using the
    trained Random Forest model.
    """
)

# Forecast year selector
forecast_end_year = st.selectbox(
    "Forecast through year",
    options=[2026, 2027, 2028, 2029, 2030],
    index=4,
    key="forecast_end_year"
)

future_years = list(range(2026, forecast_end_year + 1))

# Generate forecasts
with st.spinner("Generating future rainfall forecast..."):

    forecast_rows = []

    for forecast_year in future_years:

        result = predictor.predict(
            pred_selected_state,
            pred_selected_district,
            forecast_year
        )

        forecast_rows.append({
            "Year": forecast_year,
            "Predicted Rainfall (mm)": result["predicted_annual"],
            "Normal Rainfall (mm)": result["district_normal"],
            "Deviation (%)": result["pct_deviation"],
            "Category": result["category"]
        })

forecast_df = pd.DataFrame(forecast_rows)

# ------------------------------------------------------------
# FORECAST SUMMARY
# ------------------------------------------------------------

fc1, fc2, fc3 = st.columns(3)

with fc1:

    first_forecast = forecast_df.iloc[0]

    st.metric(
        f"{int(first_forecast['Year'])} Forecast",
        f"{first_forecast['Predicted Rainfall (mm)']:,.1f} mm"
    )

with fc2:

    last_forecast = forecast_df.iloc[-1]

    st.metric(
        f"{int(last_forecast['Year'])} Forecast",
        f"{last_forecast['Predicted Rainfall (mm)']:,.1f} mm",
        delta=f"{last_forecast['Deviation (%)']:+.1f}% vs Normal"
    )

with fc3:

    category_counts = forecast_df["Category"].value_counts()

    dominant_category = category_counts.index[0]

    st.metric(
        "Most Frequent Category",
        dominant_category,
        delta=f"{int(category_counts.iloc[0])} of {len(forecast_df)} years"
    )

# ------------------------------------------------------------
# FORECAST TABLE
# ------------------------------------------------------------

st.markdown("##### 📋 Year-by-Year Forecast")

st.dataframe(
    forecast_df.style.format({
        "Predicted Rainfall (mm)": "{:,.1f}",
        "Normal Rainfall (mm)": "{:,.1f}",
        "Deviation (%)": "{:+.1f}%"
    }),
    use_container_width=True,
    hide_index=True
)

# ------------------------------------------------------------
# FUTURE FORECAST CHART
# ------------------------------------------------------------

st.markdown("##### 📈 Future Rainfall Projection")

fig_forecast = px.line(
    forecast_df,
    x="Year",
    y=[
        "Predicted Rainfall (mm)",
        "Normal Rainfall (mm)"
    ],
    markers=True,
    title=(
        f"Future Rainfall Forecast — "
        f"{pred_selected_district}, {pred_selected_state}"
    ),
    labels={
        "value": "Rainfall (mm)",
        "Year": "Year",
        "variable": "Legend"
    }
)

# Improve chart appearance
fig_forecast.update_traces(
    line=dict(width=3),
    marker=dict(size=9)
)

fig_forecast.update_layout(
    height=450,
    hovermode="x unified",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ),
    xaxis=dict(
        dtick=1,
        title="Year"
    ),
    yaxis=dict(
        title="Annual Rainfall (mm)"
    )
)

st.plotly_chart(
    fig_forecast,
    use_container_width=True
)

# ------------------------------------------------------------
# FORECAST INTERPRETATION
# ------------------------------------------------------------

st.markdown("##### 💡 Forecast Interpretation")

avg_forecast = forecast_df["Predicted Rainfall (mm)"].mean()
normal_value = forecast_df["Normal Rainfall (mm)"].iloc[0]
avg_deviation = ((avg_forecast - normal_value) / normal_value) * 100

if avg_deviation < -19:

    st.warning(
        f"🟠 **Deficient Rainfall Outlook:** "
        f"The average predicted rainfall through {forecast_end_year} "
        f"is {avg_forecast:.1f} mm, which is {avg_deviation:+.1f}% "
        f"below the district normal of {normal_value:.1f} mm."
    )

elif avg_deviation > 19:

    st.info(
        f"🔵 **Excess Rainfall Outlook:** "
        f"The average predicted rainfall through {forecast_end_year} "
        f"is {avg_forecast:.1f} mm, which is {avg_deviation:+.1f}% "
        f"above the district normal of {normal_value:.1f} mm."
    )

else:

    st.success(
        f"🟢 **Normal Rainfall Outlook:** "
        f"The average predicted rainfall through {forecast_end_year} "
        f"is {avg_forecast:.1f} mm, representing a "
        f"{avg_deviation:+.1f}% deviation from the district normal "
        f"of {normal_value:.1f} mm."
    )

st.caption(
    "Forecast values are Machine Learning estimates based on historical "
    "rainfall data and should be interpreted as analytical projections, "
    "not official meteorological forecasts."
)
# ------------------------------------------------------------
# DATA QUALITY
# ------------------------------------------------------------

st.divider()

st.header("🔎 Data Quality")

q1, q2 = st.columns(2)

with q1:
    st.subheader("Historical Dataset")
    st.write(f"Rows: {len(df1):,}")
    st.write(f"Columns: {len(df1.columns)}")
    st.write(
        f"Missing cells: {int(df1.isna().sum().sum())}"
    )
    st.write(
        f"Duplicate rows: {int(df1.duplicated().sum())}"
    )

with q2:
    st.subheader("District Dataset")
    st.write(f"Rows: {len(df2):,}")
    st.write(f"Columns: {len(df2.columns)}")
    st.write(
        f"Missing cells: {int(df2.isna().sum().sum())}"
    )
    st.write(
        f"Duplicate rows: {int(df2.duplicated().sum())}"
    )

st.divider()

st.caption(
    "India Rainfall Prediction & Analysis Dashboard | "
    "Historical rainfall data 1901–2015"
)
