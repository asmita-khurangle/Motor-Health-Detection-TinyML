import streamlit as st
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Motor Health Detection Using TinyML",
    page_icon="⚙️",
    layout="wide"
)

# --------------------------------------------------
# FILE LOCATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent


def find_file(filename):
    possible_paths = [
        BASE_DIR / filename,
        BASE_DIR / "results" / filename,
        BASE_DIR.parent / filename,
        BASE_DIR.parent / "results" / filename
    ]

    for path in possible_paths:
        if path.exists():
            return path

    return None


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

maintenance_path = find_file("maintenance_decisions.csv")
final_results_path = find_file("final_results.csv")
tinyml_path = find_file("tinyml_optimization.csv")


if maintenance_path is None:
    st.error("maintenance_decisions.csv was not found in the GitHub repository.")
    st.stop()

if final_results_path is None:
    st.error("final_results.csv was not found in the GitHub repository.")
    st.stop()

if tinyml_path is None:
    st.error("tinyml_optimization.csv was not found in the GitHub repository.")
    st.stop()


df = pd.read_csv(maintenance_path)
final_results = pd.read_csv(final_results_path)
tinyml_results = pd.read_csv(tinyml_path)


# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("⚙️ Motor Health Detection Using TinyML")

st.markdown(
    """
    **CWRU Bearing Dataset | Random Forest | TinyML Optimization |
    Condition-Aware Explainable Maintenance Decision Layer**
    """
)

st.divider()


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

def get_result(name, default=0):
    row = final_results[final_results["Metric"] == name]

    if not row.empty:
        return row.iloc[0]["Value"]

    return default


accuracy = get_result("Accuracy")
precision = get_result("Precision (Weighted)")
recall = get_result("Recall (Weighted)")
f1 = get_result("F1-Score (Weighted)")


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.subheader("📊 Model Performance")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", f"{float(accuracy) * 100:.2f}%")

with col2:
    st.metric("Precision", f"{float(precision) * 100:.2f}%")

with col3:
    st.metric("Recall", f"{float(recall) * 100:.2f}%")

with col4:
    st.metric("F1 Score", f"{float(f1) * 100:.2f}%")

st.divider()


# --------------------------------------------------
# DATASET SUMMARY
# --------------------------------------------------

st.subheader("📁 Dataset Summary")

total_windows = len(df)

condition_counts = df["Predicted Condition"].value_counts()

normal_count = condition_counts.get("Normal", 0)
bpfi_count = condition_counts.get("BPFI", 0)
bpfo_count = condition_counts.get("BPFO", 0)

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Total Windows", total_windows)

with c2:
    st.metric("Normal", normal_count)

with c3:
    st.metric("BPFI", bpfi_count)

with c4:
    st.metric("BPFO", bpfo_count)


# --------------------------------------------------
# CONDITION DISTRIBUTION
# --------------------------------------------------

st.subheader("🔍 Predicted Condition Distribution")

st.bar_chart(condition_counts)


# --------------------------------------------------
# RPM DISTRIBUTION
# --------------------------------------------------

if "RPM" in df.columns:

    st.subheader("⚙️ RPM Distribution")

    rpm_counts = df["RPM"].value_counts().sort_index()

    st.bar_chart(rpm_counts)


# --------------------------------------------------
# TINYML OPTIMIZATION
# --------------------------------------------------

st.subheader("🧠 TinyML Model Optimization")

try:

    original_size = tinyml_results.loc[
        tinyml_results["Metric"] == "Original Model Size (KB)",
        "Value"
    ].iloc[0]

    compact_size = tinyml_results.loc[
        tinyml_results["Metric"] == "Compact Model Size (KB)",
        "Value"
    ].iloc[0]

    reduction = tinyml_results.loc[
        tinyml_results["Metric"] == "Model Size Reduction (%)",
        "Value"
    ].iloc[0]

    compact_accuracy = tinyml_results.loc[
        tinyml_results["Metric"] == "Compact Model Accuracy",
        "Value"
    ].iloc[0]

    compact_f1 = tinyml_results.loc[
        tinyml_results["Metric"] == "Compact Model F1",
        "Value"
    ].iloc[0]

    t1, t2, t3, t4, t5 = st.columns(5)

    with t1:
        st.metric(
            "Original Model",
            f"{float(original_size):.2f} KB"
        )

    with t2:
        st.metric(
            "Compact Model",
            f"{float(compact_size):.2f} KB"
        )

    with t3:
        st.metric(
            "Size Reduction",
            f"{float(reduction):.2f}%"
        )

    with t4:
        st.metric(
            "Compact Accuracy",
            f"{float(compact_accuracy) * 100:.2f}%"
        )

    with t5:
        st.metric(
            "Compact F1",
            f"{float(compact_f1) * 100:.2f}%"
        )

except Exception as e:

    st.warning(
        "TinyML optimization metrics could not be displayed: "
        + str(e)
    )


st.divider()


# --------------------------------------------------
# MAINTENANCE DECISION LAYER
# --------------------------------------------------

st.subheader("🛠️ Condition-Aware Maintenance Decision Layer")

if "Predicted Condition" in df.columns:

    selected_condition = st.selectbox(
        "Filter by predicted condition",
        ["All"] + sorted(df["Predicted Condition"].dropna().unique().tolist())
    )

    filtered_df = df.copy()

    if selected_condition != "All":
        filtered_df = filtered_df[
            filtered_df["Predicted Condition"] == selected_condition
        ]

else:

    filtered_df = df


# --------------------------------------------------
# CONFIDENCE / STATUS
# --------------------------------------------------

if "Confidence (%)" in df.columns:

    avg_confidence = df["Confidence (%)"].mean()

    st.metric(
        "Average Prediction Confidence",
        f"{avg_confidence:.2f}%"
    )


if "Status" in df.columns:

    status_counts = df["Status"].value_counts()

    st.write("### Status Distribution")

    st.bar_chart(status_counts)


# --------------------------------------------------
# MAINTENANCE RECORDS
# --------------------------------------------------

st.write("### Maintenance Decision Records")

st.dataframe(
    filtered_df,
    use_container_width=True
)


# --------------------------------------------------
# DOWNLOAD
# --------------------------------------------------

csv_data = filtered_df.to_csv(index=False)

st.download_button(
    label="⬇️ Download Maintenance Decisions CSV",
    data=csv_data,
    file_name="maintenance_decisions_filtered.csv",
    mime="text/csv"
)


# --------------------------------------------------
# IMPLEMENTATION SCOPE
# --------------------------------------------------

st.divider()

st.subheader("📌 Current Implementation Scope")

st.markdown(
    """
    **Implemented CSE/software components:**

    - CWRU bearing vibration dataset processing
    - Signal windowing and feature extraction
    - Normal / BPFI / BPFO classification
    - Random Forest machine-learning model
    - File-wise model evaluation
    - TinyML-oriented model optimization
    - Condition-Aware Explainable Maintenance Decision Layer
    - Interactive Streamlit dashboard

    **Current dataset scope:**

    The present validation uses vibration data from the CWRU bearing dataset.
    Current/temperature/load sensor fusion and ESP32 hardware integration
    are planned for the next implementation stage.
    """
)


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Motor Health Detection Using TinyML | CWRU Bearing Dataset | "
    "CSE Software Implementation"
)