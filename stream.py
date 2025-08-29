import streamlit as st
import numpy as np
import pandas as pd
import joblib
import base64
from tensorflow.keras.models import load_model
import pickle

st.set_page_config(page_title="Tool Wear Prediction", page_icon="🛠", layout="wide")

def set_background():
    
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f5f5f5;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

set_background()



# 🔷 Load models
@st.cache_resource
def load_models():
    tool_model = load_model("model.h5")
    scaler = joblib.load("scaler.joblib")
    return tool_model, scaler

tool_model,scaler = load_models()

scaler = joblib.load("scaler.joblib")
with open("scaler_columns.pkl", "rb") as f:
    expected_columns = pickle.load(f)


# 🔷 Features used
top_feature_cols = [
    'No', 'Y1_OutputCurrent', 'clamp_pressure', 'M1_CURRENT_FEEDRATE',
    'X1_OutputCurrent', 'feedrate', 'X1_CommandPosition',
    'X1_ActualPosition', 'Y1_CommandPosition', 'X1_OutputVoltage',
    'Y1_OutputVoltage', 'Z1_CommandPosition', 'S1_OutputCurrent',
    'X1_CurrentFeedback', 'M1_CURRENT_PROGRAM_NUMBER', 'Y1_ActualVelocity',
    'Y1_CommandVelocity', 'S1_ActualAcceleration'
]
user_features = [f for f in top_feature_cols if f != 'No']
sequence_length = 10

# 🔷 Feature guidance
feature_guidance = {
    'clamp_pressure': {'Worn': '3–4', 'Unworn': '7–9'},
    'feedrate': {'Worn': '3–10', 'Unworn': '90–140'},
    'M1_CURRENT_FEEDRATE': {'Worn': '3–20', 'Unworn': '90–140'},
    'Y1_OutputCurrent': {'Worn': '320–330', 'Unworn': '90–150'},
}

# 🔷 Sidebar styling
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background: linear-gradient(to bottom, #0f172a, #1d4ed8, #06b6d4);
        color: white;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] label {
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# 🔷 Sidebar navigation
menu = ["Home", "Predict"]
choice = st.sidebar.selectbox("Navigation", menu)

# 🔷 Home Page
if choice == "Home":
    st.title("🔨 CNC Tool Wear Prediction")
    st.markdown("""
    <h3 style='color: green;'>About this App 🛠</h3>
    <p>This app predicts <b>Tool Condition (Worn/Unworn)</b>, <b>Machining Finalized</b>, and <b>Visual Inspection</b> status based on CNC sensor data.</p>
    
    ### 🌍 Why Use This?
    - ✅ Reduce downtime
    - 💰 Save costs on tool replacements
    - 🎯 Improve machining quality

    📌 Powered by Deep Learning (1D CNN) for sequential sensor data analysis.
    """, unsafe_allow_html=True)

# 🔷 Prediction Page
elif choice == "Predict":
    st.title("🎛 Tool Wear Prediction Panel ⚙️")

    st.sidebar.header("📊 Feature Input & Guidance")
    with st.sidebar.expander("ℹ️ Feature Ranges"):
        for feat, vals in feature_guidance.items():
            st.write(f"**{feat}** — Worn: {vals['Worn']} | Unworn: {vals['Unworn']}")

    uploaded_file = st.sidebar.file_uploader("📎 Upload CNC CSV File", type=["csv"])
    input_df = None

    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if 'No' not in df.columns:
                df['No'] = 0.0
            missing = [col for col in top_feature_cols if col not in df.columns]
            if missing:
                st.error(f"❌ Missing columns: {', '.join(missing)}")
            else:
                input_df = df[top_feature_cols].iloc[:sequence_length]
                if len(input_df) < sequence_length:
                    pad = pd.DataFrame([input_df.iloc[-1]] * (sequence_length - len(input_df)))
                    input_df = pd.concat([input_df, pad], ignore_index=True)
        except Exception as e:
            st.error(f"❗ Error: {e}")

    else:
        st.sidebar.markdown("👈 Or enter values manually:")
        data = {f: st.sidebar.number_input(f, value=0.0) for f in user_features}
        data['No'] = 0.0
        input_df = pd.DataFrame([data] * sequence_length)

    # 🔷 Predict Button
    if st.sidebar.button("🚀 Predict") and input_df is not None:
        if (input_df['clamp_pressure'] < 6).any():
            st.error("⚠️ Warning: Clamping Pressure too low! Risk of faulty parts.")

                # Ensure all required columns are present
        for col in top_feature_cols:
            if col not in input_df.columns:
                input_df[col] = 0.0  # Fill with zeros or appropriate defaults

        # Align columns before scaling
        input_df = input_df.reindex(columns=expected_columns).fillna(0)

        # Check shape of input before reshaping
        scaled = scaler.transform(input_df)

    # Check if input is empty
        if scaled.shape[0] == 0:
            st.error("⚠️ Error: No input data provided for prediction.")
            st.stop()

    # Check if feature size is correct
        expected_features = 49
        if scaled.shape[1] != expected_features:
            st.error(f"⚠️ Error: Model expects {expected_features} features, but got {scaled.shape[1]}.")
            st.stop()

    # Reshape to match model input (1, sequence_length, 49)
        reshaped = np.reshape(scaled, (1, scaled.shape[0], scaled.shape[1]))

    # Now safe to predict
        tool_prob = tool_model.predict(reshaped, verbose=0)[0][0]
        tool_class = int(tool_prob > 0.4)
        tool_label = 'Worn' if tool_class else 'Unworn'
        confidence = tool_prob * 100 if tool_class else (1 - tool_prob) * 100

        # 🔄 Dynamic conditions based on experiment data
        if input_df['clamp_pressure'].mean() > 6 and input_df['feedrate'].mean() > 80:
            machining = "Yes"
        else:
            machining = "No"   


         # 🔷 Output Results
        st.subheader("📝 Prediction Results")
        st.success(f"🔧 Tool Condition: {tool_label} ({confidence:.2f}%)")
        st.info(f"🏭 Machining Finalized: {machining}")

            # 🔷 Export Report
        report = {
            "Tool Condition": tool_label,
            "Confidence": f"{confidence:.2f}%",
            "Machining Finalized": machining,
        
            }
        st.download_button("⬇️ Download CSV", pd.DataFrame([report]).to_csv(index=False), file_name="report.csv")

        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_fill_color(13, 78, 216)
        pdf.set_text_color(255, 255, 255)            
        pdf.cell(0, 12, " CNC Tool Wear Prediction Report ", ln=True, align='C', fill=True)
        pdf.ln(10)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 10, f"Tool Condition: {tool_label} ({confidence:.2f}%)", ln=True)
        pdf.output("report.pdf")

        with open("report.pdf", "rb") as f:
            st.download_button("⬇️ Download PDF Report", f.read(), file_name="tool_wear_report.pdf")


