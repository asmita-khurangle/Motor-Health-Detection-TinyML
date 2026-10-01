import streamlit as st
import pandas as pd
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Motor Health Detection Using TinyML",
    page_icon="⚙️",
    layout="wide"
)


# ============================================================
# FILE LOCATION HANDLING
# Works both locally and on Streamlit Cloud
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


def find_file(filename):
    possible_paths = [
        BASE_DIR / filename,
        BASE_DIR / "results" / filename,
        BASE_DIR.parent / filename,
        BASE_DIR.parent / "results" / filename,
    ]

    for path in possible_paths:
        if path.exists():
            return path

    return None


# ============================================================
# LOAD DATA
# ============================================================

maintenance_file = find_file("maintenance_decisions.csv")
final_results_file = find_file("final_results.csv")
tinyml_file = find_file("tinyml_optimization.csv")


if maintenance_file is None:
    st.error("maintenance_decisions.csv was not found.")
    st.stop()

if final_results_file is None:
    st.error("final_results.csv was not found.")
    st.stop()

if tinyml_file is None:
    st.error("tinyml_optimization.csv was not found.")
    st.stop()


maintenance_df = pd.read_csv(maintenance_file)
final_results_df = pd.read_csv(final_results_file)
tinyml_df = pd.read_csv(tinyml_file)


# ============================================================
# TITLE
# ============================================================

st.title("⚙️ Motor Health Detection Using TinyML")

st.subheader("CWRU Bearing Dataset – Vibration-Based Motor Health Detection")

st.write(
    """
This dashboard presents the implemented CSE/software part of the project.
The system processes vibration signals from the CWRU bearing dataset,
extracts time-domain and frequency-domain features, and uses a Random
Forest classifier to identify motor bearing conditions.
"""
)


# ============================================================
# IMPORTANT EVALUATION NOTE
# ============================================================

st.info(
    """
**Evaluation basis:** 428 windows from the CWRU bearing dataset.

The reported performance metrics represent the evaluated dataset/windows.
They should not be interpreted as 100% accuracy on all real-world industrial motors.
"""
)


# ============================================================
# HELPER FUNCTION
# ============================================================

def get_result_value(label):
    rows = final_results_df[
        final_results_df.iloc[:, 0].astype(str).str.strip().str.lower()
        == label.lower()
    ]

    if not rows.empty:
        return rows.iloc[0, 1]

    return None


# ============================================================
# MODEL PERFORMANCE
# ============================================================

st.header("📊 Model Performance")

accuracy = get_result_value("Accuracy")
precision = get_result_value("Precision (Weighted)")
recall = get_result_value("Recall (Weighted)")
f1 = get_result_value("F1-Score (Weighted)")


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Accuracy",
        f"{float(accuracy) * 100:.0f}%" if accuracy is not None and float(accuracy) <= 1 else f"{float(accuracy):.0f}%"
    )

with col2:
    st.metric(
        "Precision",
        f"{float(precision) * 100:.0f}%" if precision is not None and float(precision) <= 1 else f"{float(precision):.0f}%"
    )

with col3:
    st.metric(
        "Recall",
        f"{float(recall) * 100:.0f}%" if recall is not None and float(recall) <= 1 else f"{float(recall):.0f}%"
    )

with col4:
    st.metric(
        "F1-Score",
        f"{float(f1) * 100:.0f}%" if f1 is not None and float(f1) <= 1 else f"{float(f1):.0f}%"
    )


st.caption(
    "These performance metrics are calculated from the evaluated CWRU dataset windows."
)


# ============================================================
# DATASET INFORMATION
# ============================================================

st.header("📁 Dataset Information")

total_windows = len(maintenance_df)

normal_count = len(
    maintenance_df[
        maintenance_df["Actual Condition"].astype(str).str.upper() == "NORMAL"
    ]
)

bpfi_count = len(
    maintenance_df[
        maintenance_df["Actual Condition"].astype(str).str.upper() == "BPFI"
    ]
)

bpfo_count = len(
    maintenance_df[
        maintenance_df["Actual Condition"].astype(str).str.upper() == "BPFO"
    ]
)


d1, d2, d3, d4 = st.columns(4)

with d1:
    st.metric("Evaluated Windows", total_windows)

with d2:
    st.metric("Normal", normal_count)

with d3:
    st.metric("BPFI", bpfi_count)

with d4:
    st.metric("BPFO", bpfo_count)


# ============================================================
# CONDITION DISTRIBUTION
# ============================================================

st.header("🔍 Predicted Condition Distribution")

condition_counts = (
    maintenance_df["Predicted Condition"]
    .value_counts()
    .rename_axis("Condition")
    .reset_index(name="Count")
)

st.bar_chart(
    condition_counts.set_index("Condition")
)


# ============================================================
# RPM DISTRIBUTION
# ============================================================

if "RPM" in maintenance_df.columns:

    st.header("⚡ Operating Speed Distribution")

    rpm_counts = (
        maintenance_df["RPM"]
        .value_counts()
        .sort_index()
        .rename_axis("RPM")
        .reset_index(name="Windows")
    )

    st.line_chart(
        rpm_counts.set_index("RPM")
    )


# ============================================================
# TINYML OPTIMIZATION
# ============================================================

st.header("🧠 TinyML Model Optimization")

try:

    original_size = float(
        tinyml_df["Original Model Size (KB)"].iloc[0]
    )

    compact_size = float(
        tinyml_df["Compact Model Size (KB)"].iloc[0]
    )

    size_reduction = float(
        tinyml_df["Size Reduction (%)"].iloc[0]
    )

    compact_accuracy = float(
        tinyml_df["Compact Accuracy"].iloc[0]
    )

    compact_f1 = float(
        tinyml_df["Compact F1"].iloc[0]
    )

    t1, t2, t3, t4, t5 = st.columns(5)

    with t1:
        st.metric(
            "Original Model",
            f"{original_size:.2f} KB"
        )

    with t2:
        st.metric(
            "Compact Model",
            f"{compact_size:.2f} KB"
        )

    with t3:
        st.metric(
            "Size Reduction",
            f"{size_reduction:.2f}%"
        )

    with t4:
        st.metric(
            "Compact Accuracy",
            f"{compact_accuracy * 100:.0f}%"
            if compact_accuracy <= 1
            else f"{compact_accuracy:.0f}%"
        )

    with t5:
        st.metric(
            "Compact F1",
            f"{compact_f1 * 100:.0f}%"
            if compact_f1 <= 1
            else f"{compact_f1:.0f}%"
        )

except Exception:
    st.warning("TinyML optimization information could not be displayed.")


st.caption(
    "The reported model-size values refer to the implemented Python model files; "
    "final ESP32 firmware/memory usage requires hardware deployment."
)


# ============================================================
# MAINTENANCE DECISION SUMMARY
# ============================================================

st.header("🛠️ Condition-Aware Maintenance Decision")

average_confidence = maintenance_df["Confidence (%)"].mean()

warning_count = len(
    maintenance_df[
        maintenance_df["Status"].astype(str).str.upper() == "WARNING"
    ]
)

normal_status_count = len(
    maintenance_df[
        maintenance_df["Status"].astype(str).str.upper() == "NORMAL"
    ]
)


m1, m2, m3 = st.columns(3)

with m1:
    st.metric(
        "Average Confidence",
        f"{average_confidence:.2f}%"
    )

with m2:
    st.metric(
        "Warning Windows",
        warning_count
    )

with m3:
    st.metric(
        "Normal Status Windows",
        normal_status_count
    )


# ============================================================
# CONDITION FILTER
# ============================================================

st.header("📋 Maintenance Decision Records")

conditions = ["All"] + sorted(
    maintenance_df["Predicted Condition"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_condition = st.selectbox(
    "Filter by predicted condition",
    conditions
)


if selected_condition == "All":

    filtered_df = maintenance_df.copy()

else:

    filtered_df = maintenance_df[
        maintenance_df["Predicted Condition"].astype(str)
        == selected_condition
    ].copy()


# ============================================================
# DISPLAY TABLE
# ============================================================

st.dataframe(
    filtered_df,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# DOWNLOAD DATA
# ============================================================

st.download_button(
    label="⬇️ Download Maintenance Decisions CSV",
    data=maintenance_df.to_csv(index=False),
    file_name="maintenance_decisions.csv",
    mime="text/csv"
)


# ============================================================
# INNOVATION
# ============================================================

st.header("💡 Project Innovation")

st.subheader(
    "Condition-Aware Explainable Maintenance Decision Layer"
)

st.write(
    """
The innovation adds an explainable maintenance-decision layer on top of
fault classification. Instead of showing only the detected fault, the
system presents the predicted condition together with confidence,
operating speed and a recommended maintenance action.
"""
)

st.write(
    """
Example decision information includes:

• Detected condition  
• Prediction confidence  
• Operating RPM  
• Motor status  
• Evidence from vibration features  
• Recommended maintenance action
"""
)


# ============================================================
# CURRENT IMPLEMENTATION SCOPE
# ============================================================

st.header("🔧 Current Implementation Scope")

st.write(
    """
### Completed CSE/Software Work

✓ CWRU bearing dataset processing  
✓ Vibration signal preprocessing  
✓ Feature extraction  
✓ Windowing  
✓ Normal / BPFI / BPFO classification  
✓ Random Forest model  
✓ File-wise evaluation  
✓ Performance analysis  
✓ TinyML model optimization  
✓ Condition-Aware Explainable Maintenance Decision Layer  
✓ Interactive Streamlit dashboard
"""
)


# ============================================================
# LIMITATIONS AND FUTURE WORK
# ============================================================

st.header("🚀 Future Work")

st.write(
    """
The next stage is hardware and IoT integration:

• ESP32 implementation  
• Real-time vibration sensor acquisition  
• Motor hardware testing  
• Additional sensor integration such as current and temperature  
• Embedded TinyML deployment  
• MQTT communication  
• Real-time IoT monitoring  
• Validation under different loads and speeds
"""
)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Motor Health Detection Using TinyML | CWRU Bearing Dataset | "
    "CSE Software Implementation"
)