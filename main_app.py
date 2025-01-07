import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained Liver Disease model
with open("liver_model.pkl", "rb") as file:
    liver_model = pickle.load(file)

with open("parkinsons_model.pkl", "rb") as file:
    parkinsons_model = pickle.load(file)

with open("kidney_model.pkl", "rb") as file:
    kidney_model = pickle.load(file)


# Streamlit dashboard
st.header("🩺MULTIPLE DISEASE PREDICTION SYSTEM")

   
st.sidebar.markdown("### Disease Prediction Options")
selected_dataset = st.sidebar.selectbox(
    "Choose a Disease to Predict:",
    ("🏥 Home", "🧠 Parkinson's Disease", "🫁 Liver Disease", "🫘 Kidney Disease")
)

if selected_dataset == "🏥 Home":
    st.write("""
            Welcome to the **Multiple Disease Prediction System**. This application allows you to predict the likelihood of certain diseases based on input medical data. 
            Currently, the system supports predictions for the following conditions:
            - **🧠 Parkinson's Disease**
            - **🫁 Liver Disease**
            - **🫘 Kidney Disease** 

            To get started:
            1. Use the sidebar to select a disease you want to predict.
            2. Enter the required medical measurements in the input fields.
            3. Click on the "Predict" button to see the results.

            **Note:** This tool is for educational purposes and should not be used as a substitute for professional medical advice.
            """)

elif selected_dataset == "🧠 Parkinson's Disease":
    st.header("🧠 Parkinson's Disease Prediction")
     # Input fields for Parkinson's disease prediction
    MDVP_Fo_Hz = st.number_input("Fundamental Frequency (MDVP:Fo(Hz))", min_value=0.0, value=0.0)
    MDVP_Fhi_Hz = st.number_input("Maximum Frequency (MDVP:Fhi(Hz))", min_value=0.0, value=0.0)
    MDVP_Flo_Hz = st.number_input("Minimum Frequency (MDVP:Flo(Hz))", min_value=0.0, value=0.0)
    MDVP_Jitter_percent = st.number_input("Jitter (MDVP:Jitter(%))", min_value=0.0, value=0.0)
    MDVP_Jitter_Abs = st.number_input("Absolute Jitter (MDVP:Jitter(Abs))", min_value=0.0, value=0.0)
    MDVP_RAP = st.number_input("Relative Average Perturbation (MDVP:RAP)", min_value=0.0, value=0.0)
    MDVP_PPQ = st.number_input("Pitch Period Perturbation Quotient (MDVP:PPQ)", min_value=0.0, value=0.0)
    Jitter_DDP = st.number_input("Degree of Derivative Perturbation (Jitter:DDP)", min_value=0.0, value=0.0)
    MDVP_Shimmer = st.number_input("Shimmer (MDVP:Shimmer)", min_value=0.0, value=0.0)
    MDVP_Shimmer_dB = st.number_input("Shimmer in dB (MDVP:Shimmer(dB))", min_value=0.0, value=0.0)
    Shimmer_APQ3 = st.number_input("Amplitude Perturbation Quotient (Shimmer:APQ3)", min_value=0.0, value=0.0)
    Shimmer_APQ5 = st.number_input("Amplitude Perturbation Quotient (Shimmer:APQ5)", min_value=0.0, value=0.0)
    MDVP_APQ = st.number_input("Amplitude Perturbation Quotient (MDVP:APQ)", min_value=0.0, value=0.0)
    Shimmer_DDA = st.number_input("Difference of Average Amplitude (Shimmer:DDA)", min_value=0.0, value=0.0)
    NHR = st.number_input("Noise-to-Harmonics Ratio (NHR)", min_value=0.0, value=0.0)
    HNR = st.number_input("Harmonics-to-Noise Ratio (HNR)", min_value=0.0, value=0.0)
    RPDE = st.number_input("Recurrence Period Density Entropy (RPDE)", min_value=0.0, value=0.0)
    DFA = st.number_input("Detrended Fluctuation Analysis (DFA)", min_value=0.0, value=0.0)
    spread1 = st.number_input("Signal Spread 1 (spread1)", value=0.0)
    spread2 = st.number_input("Signal Spread 2 (spread2)", value=0.0)
    D2 = st.number_input("Correlation Dimension (D2)", min_value=0.0, value=0.0)
    PPE = st.number_input("Pitch Period Entropy (PPE)", min_value=0.0, value=0.0)

    # Prepare input features as a 2D array for prediction
    input_features = np.array([[MDVP_Fo_Hz, MDVP_Fhi_Hz, MDVP_Flo_Hz, MDVP_Jitter_percent, MDVP_Jitter_Abs,
                                 MDVP_RAP, MDVP_PPQ, Jitter_DDP, MDVP_Shimmer, MDVP_Shimmer_dB,
                                 Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ, Shimmer_DDA, NHR, HNR,
                                 RPDE, DFA, spread1, spread2, D2, PPE]])
    

    #for col in range(input_features.shape[1]):
        #input_features[:, col] = [str(x).encode('utf-8').decode('utf-8') if isinstance(x, str) else x for x in input_features[:, col]]

    # Predict when button is clicked
    if st.button("Predict Parkinsons Disease"):
        try:
            prediction = parkinsons_model.predict(input_features)
            if prediction[0] == 1:
                st.success("🚨 Positive for Parkinson's disease.")
            else:
                st.success("✅ Negative for Parkinson's disease.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

elif selected_dataset == "🫁 Liver Disease":
    st.header("🫁 Liver Disease Prediction")
    
    # Feature Inputs for Liver Disease
    Age = st.number_input("Age", min_value=1, step=1)
    Total_Bilirubin = st.number_input("Total Bilirubin", min_value=0.0)
    Direct_Bilirubin = st.number_input("Direct Bilirubin", min_value=0.0)
    Alkaline_Phosphotase = st.number_input("Alkaline Phosphotase", min_value=0.0)
    Alamine_Aminotransferase = st.number_input("Alamine Aminotransferase", min_value=0.0)
    Aspartate_Aminotransferase = st.number_input("Aspartate Aminotransferase", min_value=0.0)
    Total_Proteins = st.number_input("Total Proteins", min_value=0.0)
    Albumin = st.number_input("Albumin", min_value=0.0)
    Albumin_and_Globulin_Ratio = st.number_input("Albumin and Globulin Ratio", min_value=0.0)

    # Prepare the input features
    input_features = np.array([[Age, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase,
                                Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Proteins,
                                Albumin, Albumin_and_Globulin_Ratio]])
    
    # Adjust input to match model's expected feature count
    expected_features = liver_model.feature_names_in_  # Get expected feature names
    input_df = pd.DataFrame(input_features, columns=expected_features[:input_features.shape[1]])

    # Add missing features with default values (e.g., 0)
    for feature in expected_features:
        if feature not in input_df.columns:
            input_df[feature] = 0  # Default value for missing features
    
    # Reorder columns to match the model's feature order
    input_df = input_df[expected_features]

    # Button for prediction
    if st.button("Predict Liver Disease"):
        try:
            prediction = liver_model.predict(input_df)  # Pass the DataFrame
            if prediction[0] == 1:
                st.success("🚨 Positive for Liver disease.")
            else:
                st.success("✅ Negative for Liver disease.")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

elif selected_dataset == "🫘 Kidney Disease":
    st.header("🫘 Kidney Disease Prediction")
    # Feature Inputs for Kidney Disease
    age = st.number_input("Age", min_value=1, step=1)
    bp = st.number_input("Blood Pressure", min_value=0)
    sg = st.number_input("Specific Gravity", min_value=1.0, max_value=1.030, step=0.001)
    al = st.number_input("Albumin", min_value=0.0)
    su = st.number_input("Sugar", min_value=0.0)
    rbc = st.number_input("Red Blood Cells", min_value=0.0)
    pc = st.number_input("Pus Cells", min_value=0.0)
    sc = st.number_input("Serum Creatinine Level", min_value=0.0)
    sod = st.number_input("Sodium Level in Blood", min_value=0.0)
    wc = st.number_input("White Blood Cells", min_value=0.0)
    rc = st.number_input("Red Blood Cell Count", min_value=0.0)
    cad = st.number_input("Coronary Artery Disease", min_value=0.0)
    appet = st.number_input("Appetite Status", min_value=0.0)

    # Prepare the input features
    input_features = np.array([[age, bp, sg, al, su, rbc, pc, sc, sod, wc, rc, cad, appet]])

    # Adjust input features to match model's expected feature count
    expected_features = kidney_model.n_features_in_
    if input_features.shape[1] < expected_features:
        # Add missing features with a default value (e.g., 0)
        missing_features = expected_features - input_features.shape[1]
        input_features = np.hstack([input_features, np.zeros((input_features.shape[0], missing_features))])
    elif input_features.shape[1] > expected_features:
        # Trim extra features
        input_features = input_features[:, :expected_features]

    # Predict when the button is clicked
    if st.button("Predict Kidney Disease"):
        try:
            prediction = kidney_model.predict(input_features)
            if prediction[0] == 1:
                st.success("🚨 Positive for Kidney Disease")
            else:
                st.success("✅ Negative for Kidney Disease")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

         

