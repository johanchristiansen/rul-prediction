# streamlit_app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.tree import plot_tree
from treeinterpreter import treeinterpreter as ti
import lime
import lime.lime_tabular
import shap
from IPython.display import HTML
from streamlit.components.v1 import html
import huggingface_hub

# Set up directories
os.makedirs("output_images", exist_ok=True)

# Load model and scaler
@st.cache_resource
def load_model():
    try:
        local_model_path = "random_forest_model_compressed.pkl"
        if os.path.exists(local_model_path):
            model = joblib.load(local_model_path)
        else:
            token = st.secrets.get("HF_TOKEN", None)
            # Download model from Hugging Face
            model_path = huggingface_hub.hf_hub_download(
                repo_id="johanchristiansen/rul_optimized",
                filename="random_forest_model_compressed.pkl",
                token=token  # Add token if needed
            )
            model = joblib.load(model_path)
        scaler = joblib.load('scaler.pkl')
        return model, scaler
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Wrap the model loading in try-except
try:
    rf_model, scaler = load_model()
    if rf_model is None or scaler is None:
        st.error("Failed to load model or scaler. Please check the logs.")
        st.stop()
except Exception as e:
    st.error(f"Application startup error: {str(e)}")
    st.stop()

# Input widgets
st.sidebar.header("Input Parameters")
cycle = st.sidebar.number_input("Cycle", min_value=0.0, value=100.0)
voltage_measured = st.sidebar.number_input("Voltage measured", value=3.7)
voltage_load = st.sidebar.number_input("Voltage load", value=3.7)
current_load = st.sidebar.number_input("Current load", value=1.0)
temperature_measured = st.sidebar.number_input("Temperature measured", value=25.0)

visualization_choice = st.sidebar.selectbox(
    "Visualization Method",
    ("Tree Interpreter", "LIME", "SHAP")
)

# Visualization functions
def visualize_local_contributions(model, scaled_input, feature_names):
    prediction, bias, contributions = ti.predict(model, scaled_input)
    total_contributions = contributions[0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(feature_names, total_contributions,
                  color=['#4CAF50' if c > 0 else '#F44336' for c in total_contributions])
    ax.set_title("Local Feature Contributions")
    ax.set_xlabel("Contribution Value")
    plt.gca().invert_yaxis()
    
    for bar in bars:
        width = bar.get_width()
        label_x = width if width > 0 else width - 0.1
        ax.text(label_x, bar.get_y() + bar.get_height()/2, f"{width:.2f}", 
                va='center', ha='left' if width > 0 else 'right')
    
    st.pyplot(fig)
    plt.close()

def visualize_with_lime(model, scaled_input, feature_names):
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=scaler.transform(pd.read_csv('discharge_df.csv')[feature_names]),
        feature_names=feature_names,
        mode='regression'
    )
    exp = explainer.explain_instance(
        scaled_input[0], 
        model.predict, 
        num_features=len(feature_names)
    )
    html = exp.as_html()
    st.components.v1.html(html, height=800)

def visualize_with_shap(model, scaled_input, feature_names):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled_input)
    
    st.subheader("SHAP Force Plot")
    shap.initjs()
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0], scaled_input[0], feature_names=feature_names)
    st.components.v1.html(shap.getjs() + force_plot.html(), height=400)
    
    st.subheader("SHAP Summary Plot")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, scaled_input, feature_names=feature_names, show=False)
    st.pyplot(fig)
    plt.close()


# Prediction and display
if st.sidebar.button("Predict RUL"):
    input_data = {
        'cycle': cycle,
        'Voltage_measured': voltage_measured,
        'Voltage_load': voltage_load,
        'Current_load': current_load,
        'Temperature_measured': temperature_measured
    }
    
    input_df = pd.DataFrame([input_data])
    scaled_input = scaler.transform(input_df)
    prediction = rf_model.predict(scaled_input)[0]
    
    st.header("Prediction Result")
    st.success(f"Predicted Remaining Useful Life (RUL): {prediction:.2f} seconds")
    
    feature_names = ['cycle', 'Voltage_measured', 'Voltage_load', 'Current_load', 'Temperature_measured']
    
    st.header("Model Explanation")
    if visualization_choice == "Tree Interpreter":
        visualize_local_contributions(rf_model, scaled_input, feature_names)
    elif visualization_choice == "LIME":
        visualize_with_lime(rf_model, scaled_input, feature_names)
    elif visualization_choice == "SHAP":
        visualize_with_shap(rf_model, scaled_input, feature_names)