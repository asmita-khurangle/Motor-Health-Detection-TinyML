import streamlit as st
import pandas as pd
import os

# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Motor Health Detection Using TinyML",
    page_icon="⚙️",
    layout="wide"
)

# --------------------------------------------------
# Paths
# --------------------------------------------------

MAINTENANCE_PATH = "results/maintenance_decisions.csv"
FINAL_RESULTS_PATH = "results/final_results.csv"
OPTIMIZATION_PATH = "results/tinyml_optimization.csv"

# --------------------------------------------------
# Load data
# --------------------------------------------------

df = pd.read_csv(MAINTENANCE_PATH)
final_results = pd.read_csv(FINAL_RESULTS_PATH)
optimization = pd.read_csv(OPTIMIZATION_PATH)

# --------------------------------------------------
# Helper function
# --------------------------------------------------

def get_metric(name):
    row = final_results[
        final_results["Metric"] == name
    ]

    if len(row) == 0:
        return 0

    return float(row.iloc[0]["Value"])


# --------------------------------------------------
# Metrics
# --------------------------------------------------

accuracy = get_metric("Accuracy") * 100
precision = get_metric("Precision (Weighted)") * 100
recall = get_metric("Recall (Weighted)") * 100
f1 = get_metric("F1-Score (Weighted)") * 100

total_windows = len(df)

average_confidence = df["Confidence (%)"].mean()

normal_count = (
    df["Predicted Condition"] == "Normal"
).sum()

bpfi_count = (
    df["Predicted Condition"] == "BPFI"
).sum()

bpfo_count = (
    df["Predicted Condition"] == "BPFO"
).sum()

# --------------------------------------------------
# Model size
# --------------------------------------------------

original_row = optimization[
    optimization["Model"] == "Original Random Forest"
].iloc[0]

compact_row = optimization[
    optimization["Model"] == "Compact Random Forest"
].iloc[0]

original_size = float(original_row["Size_KB"])
compact_size = float(compact_row["Size_KB"])

size_reduction = (
    (original_size - compact_size)
    / original_size
) * 100

compact_accuracy = float(
    compact_row["Accuracy"]
) * 100

compact_f1 = float(
    compact_row["F1_Score"]
) * 100

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("⚙️ Motor Health Detection Using TinyML")

st.write(
    "Condition-aware bearing fault detection and "
    "maintenance decision system using the CWRU dataset."
)

st.caption(
    "Software/ML validation using vibration features from the "
    "CWRU bearing dataset."
)

st.divider()

# --------------------------------------------------
# Section 1 — ML Performance
# --------------------------------------------------

st.header("📊 ML Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Accuracy",
        f"{accuracy:.2f}%"
    )

with col2:
    st.metric(
        "Precision",
        f"{precision:.2f}%"
    )

with col3:
    st.metric(
        "Recall",
        f"{recall:.2f}%"
    )

with col4:
    st.metric(
        "F1-Score",
        f"{f1:.2f}%"
    )

# --------------------------------------------------
# Section 2 — Dataset
# --------------------------------------------------

st.header("📁 Dataset Analysis")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Windows",
        total_windows
    )

with col2:
    st.metric(
        "Normal",
        normal_count
    )

with col3:
    st.metric(
        "BPFI",
        bpfi_count
    )

with col4:
    st.metric(
        "BPFO",
        bpfo_count
    )

# --------------------------------------------------
# Condition distribution
# --------------------------------------------------

st.subheader("Predicted Condition Distribution")

condition_counts = (
    df["Predicted Condition"]
    .value_counts()
)

st.bar_chart(condition_counts)

# --------------------------------------------------
# RPM
# --------------------------------------------------

st.subheader("Operating Speed Distribution")

rpm_counts = (
    df["RPM"]
    .value_counts()
    .sort_index()
)

st.bar_chart(rpm_counts)

# --------------------------------------------------
# TinyML optimization
# --------------------------------------------------

st.header("⚡ TinyML Optimization")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Original Model",
        f"{original_size:.2f} KB"
    )

with col2:
    st.metric(
        "Compact Model",
        f"{compact_size:.2f} KB"
    )

with col3:
    st.metric(
        "Size Reduction",
        f"{size_reduction:.2f}%"
    )

with col4:
    st.metric(
        "Compact Accuracy",
        f"{compact_accuracy:.2f}%"
    )

st.write(
    f"Compact model F1-score: **{compact_f1:.2f}%**"
)

# --------------------------------------------------
# Maintenance decision
# --------------------------------------------------

st.header("🔧 Condition-Aware Maintenance Decision")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Average Confidence",
        f"{average_confidence:.2f}%"
    )

with col2:
    warning_count = (
        df["Status"] == "WARNING"
    ).sum()

    st.metric(
        "Warning Windows",
        warning_count
    )

with col3:
    normal_status_count = (
        df["Status"] == "NORMAL"
    ).sum()

    st.metric(
        "Normal Windows",
        normal_status_count
    )

# --------------------------------------------------
# Filter
# --------------------------------------------------

st.subheader("🔎 Explore Maintenance Decisions")

conditions = sorted(
    df["Predicted Condition"].unique()
)

selected_condition = st.selectbox(
    "Select condition",
    ["All"] + conditions
)

if selected_condition == "All":

    filtered_df = df

else:

    filtered_df = df[
        df["Predicted Condition"]
        == selected_condition
    ]

# --------------------------------------------------
# Maintenance records
# --------------------------------------------------

st.dataframe(
    filtered_df,
    use_container_width=True,
    height=400
)

# --------------------------------------------------
# Download
# --------------------------------------------------

csv_data = filtered_df.to_csv(index=False)

st.download_button(
    label="⬇️ Download Maintenance Results",
    data=csv_data,
    file_name="maintenance_decisions.csv",
    mime="text/csv"
)

# --------------------------------------------------
# Project scope
# --------------------------------------------------

st.divider()

st.header("ℹ️ Current Implementation Scope")

st.write(
    """
    **Completed software implementation:**
    
    • CWRU vibration dataset processing  
    • Time-domain and frequency-domain feature extraction  
    • Window-based signal analysis  
    • Random Forest fault classification  
    • File-wise model evaluation  
    • TinyML model-size optimization  
    • Confidence-based inference  
    • Condition-aware maintenance decision layer  
    • Interactive Streamlit dashboard  
    
    **Current limitation:** The present ML validation uses vibration
    data from the CWRU dataset. Physical ESP32, current, temperature,
    and other sensor integration will be treated as the hardware
    integration stage rather than being represented as completed
    measurements.
    """
)

st.caption(
    "Motor Health Detection Using TinyML | CSE Software Implementation"
)