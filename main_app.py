import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# Load models
with open("parkinsons_model.pkl", "rb") as file:
    parkinsons_model = pickle.load(file)

with open("kidney_model.pkl", "rb") as file:
    kidney_model = pickle.load(file)

with open("https://github.com/Nishamisty/Multiple_Disease/blob/main/liver_model.pkl", "rb") as file:
    liver_model = pickle.load(file)

# Streamlit dashboard
st.header("MULTIPLE DISEASE PREDICTION SYSTEM")
st.subheader("Select a Disease to Predict")

selected_dataset = st.sidebar.selectbox(
    "Choose Disease to Predict", 
    ("🧠 Parkinson's Disease", "🫁 Liver Disease", "🫘 Kidney Disease")
)
if selected_dataset == "🧠 Parkinson's Disease":
    st.header("🧠 Parkinson's Disease Prediction")
    # Collect Feature Inputs
    MDVP_Fo = st.number_input("MDVP:Fo(Hz)", min_value=0.0, step=0.1)
    MDVP_Fhi = st.number_input("MDVP:Fhi(Hz)", min_value=0.0, step=0.1)
    MDVP_Flo = st.number_input("MDVP:Flo(Hz)", min_value=0.0, step=0.1)
    MDVP_Jitter = st.number_input("MDVP:Jitter(%)", min_value=0.0, step=0.001)
    MDVP_Jitter_Abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, step=0.001)
    MDVP_RAP = st.number_input("MDVP:RAP", min_value=0.0, step=0.001)
    MDVP_PPQ = st.number_input("MDVP:PPQ", min_value=0.0, step=0.001)
    MDVP_Shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, step=0.001)
    MDVP_Shimmer_DB = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, step=0.001)
    Shimmer_APQ3 = st.number_input("Shimmer:APQ3", min_value=0.0, step=0.001)
    Shimmer_APQ5 = st.number_input("Shimmer:APQ5", min_value=0.0, step=0.001)
    MDVP_APQ = st.number_input("MDVP:APQ", min_value=0.0, step=0.001)
    Shimmer_DDA = st.number_input("Shimmer:DDA", min_value=0.0, step=0.001)
    NHR = st.number_input("NHR", min_value=0.0, step=0.001)
    HNR = st.number_input("HNR", min_value=0.0, step=0.1)
    RPDE = st.number_input("RPDE", min_value=0.0, step=0.001)
    DFA = st.number_input("DFA", min_value=0.0, step=0.001)
    spread1 = st.number_input("Spread1", step=0.001)
    spread2 = st.number_input("Spread2", step=0.001)
    D2 = st.number_input("D2", min_value=0.0, step=0.001)
    PPE = st.number_input("PPE", min_value=0.0, step=0.001)

    # Create the input feature array
    input_features = np.array([[MDVP_Fo, MDVP_Fhi, MDVP_Flo, MDVP_Jitter, MDVP_Jitter_Abs, MDVP_RAP, MDVP_PPQ, 
                                MDVP_Shimmer, MDVP_Shimmer_DB, Shimmer_APQ3, Shimmer_APQ5, MDVP_APQ,
                                Shimmer_DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]])

    # Check for feature mismatch
    expected_features = parkinsons_model.n_features_in_
    if input_features.shape[1] < expected_features:
        # Add missing features (e.g., zeros for simplicity)
        missing_features = expected_features - input_features.shape[1]
        input_features = np.hstack([input_features, np.zeros((input_features.shape[0], missing_features))])
    elif input_features.shape[1] > expected_features:
        # Trim extra features
        input_features = input_features[:, :expected_features]

    # Predict when button is clicked
    if st.button("Predict Parkinson's Disease"):
        prediction = parkinsons_model.predict(input_features)
        if prediction ==0:
               st.write ("🚨 Positive for Parkinson's Disease")
        else:
               st.write("✅ Negative for Parkinson's Disease")


# Feature names that the model expects
        # Feature names that the model expects

# Streamlit input for features
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

    expected_features = liver_model.feature_names_in_

    # Check the number of features your model expects
    expected_feature_count = len(expected_features)

    # Adjust the number of features in input data to match expected feature count
    input_feature_count = input_features.shape[1]
    
    if input_feature_count < expected_feature_count:
        # Add missing features with default value 0
        missing_feature_count = expected_feature_count - input_feature_count
        input_features = np.hstack([input_features, np.zeros((input_features.shape[0], missing_feature_count))])
    elif input_feature_count > expected_feature_count:
        # Trim extra features
        input_features = input_features[:, :expected_feature_count]

    # Convert input_features to DataFrame and ensure column names match the model's feature names
    input_df = pd.DataFrame(input_features, columns=expected_features[:input_features.shape[1]])

    # Add missing features with default values (0) and reorder columns to match the model's feature order
    for feature in expected_features:
        if feature not in input_df.columns:
            input_df[feature] = 0  # Add missing feature columns with default values

    input_df = input_df[expected_features]  # Ensure columns are in the correct order

    # Predict when the button is clicked
    if st.button("Predict Liver Disease"):
        prediction = liver_model.predict(input_df)  # Pass the DataFrame
        if prediction == 0:
               st.write ("🚨 Positive for Liver Disease")
        else:
               st.write("✅ Negative for Liver Disease")

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
        prediction = kidney_model.predict(input_features)
        result = "🚨 Positive for Kidney Disease" if prediction[0] == 1 else "✅ Negative for Kidney Disease"
        st.write(f"Prediction: {result}")
