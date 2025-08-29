# CNC-Milling-Performance-Analysis-and-Fault-Detection-1-.

CNC-Milling-Performance-Analysis-and-Fault-Detection-
CNC MILLING ANALYSIS

📌 Overview
This project focuses on analyzing CNC milling operations using time-series sensor data to build machine learning models for:

Tool Wear Detection
Inadequate Clamping Detection
Machining Completion Prediction
It includes the end-to-end pipeline: data preprocessing, feature engineering, model training & evaluation, and deployment through a Streamlit application on AWS.

🎯 Problem Statement
CNC machines collect real-time sensor data across multiple axes. The aim is to predict faults such as worn tools, incomplete machining, or poor clamping using this data for preventive maintenance and quality assurance.

💼 Business Use Cases
🧰 Predictive Maintenance: Detect worn tools to avoid breakdowns.
🔍 Quality Assurance: Identify insufficient clamp pressure leading to defective parts.
📈 Process Optimization: Optimize machining parameters to reduce cycle time.
🛡️ Operational Safety: Monitor unsafe or faulty machining behavior.
🧠 Approach
Step 1: Data Understanding
train.csv: Summary-level data with labels.
experiment_01.csv to experiment_18.csv: Time-series sensor data (100ms sampling).
Step 2: Preprocessing
Handle missing and anomalous values.
Normalize numerical features.
Step 3: Feature Engineering
Statistical features (mean, std, min, max, skew) from time-series data for each experiment.
Step 4: Model Training
Train a Random Forest Classifier to detect tool wear (binary classification).
Evaluate using classification metrics.
Step 5: Deployment
Interactive Streamlit app for real-time fault prediction.
Hosted using AWS (EC2 or Streamlit Community Cloud).
🧪 Model Evaluation Metrics
✅ Accuracy
🎯 Precision / Recall / F1-Score
🧮 Confusion Matrix
📈 ROC-AUC
⏱️ Execution Time (real-time usability)
🛠️ Tech Stack
Languages: Python
Libraries: pandas, numpy, scikit-learn, streamlit
Deployment: AWS, Streamlit
Tools: Jupyter Notebooks, Git, VS Code
📁 Project Structure
