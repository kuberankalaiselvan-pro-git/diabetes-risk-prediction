import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import os
import joblib # Import joblib for saving/loading models and scalers

# --- Function to load data and train model ---
# This function will be cached by Streamlit to avoid re-running on every interaction
@st.cache_resource
def load_data_and_train_model():
    """
    Loads the diabetes dataset, preprocesses it, and trains a RandomForestClassifier.
    This function is cached to prevent re-loading and re-training on every Streamlit rerun.
    It now also handles loading/saving models and scalers from/to .pkl files.
    """
    model_path = 'random_forest_model.pkl'
    scaler_path = 'scaler.pkl'
    feature_names_path = 'feature_names.pkl' # To store feature names for consistent order

    # Adjust paths for Kaggle environment or local execution
    # For Kaggle, models might be in /kaggle/working/ or /kaggle/input/
    # For local, they should be in the same directory.
    # We'll prioritize loading, then training and saving.

    # Check if model and scaler are already saved
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(feature_names_path):
        st.info("Loading pre-trained model and scaler from .pkl files...")
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_names = joblib.load(feature_names_path)
        return model, scaler, feature_names
    else:
        st.info("Pre-trained model or scaler not found. Training a new model...")
        # Adjust path for Kaggle environment or local execution
        data_path = 'diabetes_binary_health_indicators_BRFSS2015.csv'
        if not os.path.exists(data_path):
            # Fallback for Kaggle environment if running directly from notebook export
            data_path = '/kaggle/input/diabetes-health-indicators-dataset/diabetes_binary_health_indicators_BRFSS2015.csv'
            if not os.path.exists(data_path):
                st.error(f"Error: Dataset not found at '{data_path}' or '{os.getcwd()}/diabetes_binary_health_indicators_BRFSS2015.csv'")
                st.stop() # Stop the app if data is not found

        df = pd.read_csv(data_path)

        # Define features (X) and target (y)
        X = df[['BMI', 'Age', 'GenHlth', 'HighBP']]
        y = df['Diabetes_binary']

        # Initialize and fit StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns) # Convert back to DataFrame

        # Apply SMOTE for balancing the dataset
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled_df, y)

        # Initialize and train RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_resampled, y_resampled)

        # Save the trained model, scaler, and feature names to .pkl files
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(X.columns.tolist(), feature_names_path) # Save feature names
        st.success("Model and scaler trained and saved successfully!")

        return model, scaler, X.columns.tolist() # Return feature names for input order

# --- Load model and scaler ---
model, scaler, feature_names = load_data_and_train_model()

# --- Streamlit App Layout ---
st.set_page_config(page_title="Diabetes Prediction App", layout="centered")

st.title("🩺 Diabetes Prediction App")
st.markdown("""
    This app predicts the likelihood of diabetes based on your health indicators.
    Please enter the following information:
""")

# Input fields for features
st.header("Enter Your Health Information:")

col1, col2 = st.columns(2)

with col1:
    # BMI input
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=100.0, value=25.0, step=0.1,
                          help="Your Body Mass Index. A value between 10 and 100.")

    # HighBP input (binary: 0 or 1)
    high_bp_options = {0: "No High Blood Pressure", 1: "Yes High Blood Pressure"}
    high_bp_selection = st.radio("Do you have High Blood Pressure?", list(high_bp_options.keys()), format_func=lambda x: high_bp_options[x], index=0,
                                  help="Select 'Yes' if you have high blood pressure, 'No' otherwise.")
    high_bp = float(high_bp_selection) # Ensure it's float for consistency with model

with col2:
    # Age input (categorical, 1-13)
    # Mapping from notebook: 1 = 18-24, ..., 13 = 80 or older
    age_mapping = {
        1: "18-24", 2: "25-29", 3: "30-34", 4: "35-39", 5: "40-44",
        6: "45-49", 7: "50-54", 8: "55-59", 9: "60-64", 10: "65-69",
        11: "70-74", 12: "75-79", 13: "80 or older"
    }
    age_selection = st.selectbox("Age Group", options=list(age_mapping.keys()), format_func=lambda x: age_mapping[x], index=8, # Default to 60-64
                                  help="Select your age group based on the provided categories.")
    age = float(age_selection) # Ensure it's float

    # GenHlth input (categorical, 1-5)
    # Mapping: 1 = Excellent, 2 = Very good, 3 = Good, 4 = Fair, 5 = Poor
    gen_hlth_mapping = {
        1: "Excellent", 2: "Very good", 3: "Good", 4: "Fair", 5: "Poor"
    }
    gen_hlth_selection = st.selectbox("General Health Condition", options=list(gen_hlth_mapping.keys()), format_func=lambda x: gen_hlth_mapping[x], index=2, # Default to Good
                                       help="Rate your general health condition.")
    gen_hlth = float(gen_hlth_selection) # Ensure it's float


# --- Prediction Button ---
st.markdown("---")
if st.button("Predict Diabetes Risk"):
    # Create a DataFrame from user inputs, ensuring column order matches training data
    input_data = pd.DataFrame([[bmi, age, gen_hlth, high_bp]], columns=feature_names)

    # Scale the input data using the *fitted* scaler
    input_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)[:, 1] # Probability of positive class (diabetes)

    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error(f"**Based on the provided information, the model predicts a HIGH risk of Diabetes.**")
        st.write(f"Probability of Diabetes: **{prediction_proba[0]:.2f}**")
    else:
        st.success(f"**Based on the provided information, the model predicts a LOW risk of Diabetes.**")
        st.write(f"Probability of Diabetes: **{prediction_proba[0]:.2f}**")

    st.markdown("""
    <div style="font-size: 0.8em; color: gray;">
    *Disclaimer: This prediction is based on a machine learning model and should not be considered medical advice.
    Always consult with a healthcare professional for diagnosis and treatment.*
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("Developed by Your Name/Team") # You can customize this
