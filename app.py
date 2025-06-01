# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve

# --- ADD IMAGE AT THE TOP ---
st.image("header_image.png", use_column_width=True)

# Title and introduction
st.title("🏋️‍♂️ Sport Injury Risk Predictor")
st.markdown("""
This interactive application predicts the injury risk of an athlete based on key features.  
Upload your dataset, explore the data, and get predictions for new or existing data.
""")

# Load the trained model and scaler
model = joblib.load("logistic_model_top7.pkl")

# Sidebar: Upload CSV file
st.sidebar.header("Upload Your Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Load and display the data
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Show basic info
    st.write("### Data Info")
    st.write(df.describe())

    # Display class imbalance
    if 'Injury_Indicator' in df.columns:
        st.write("### Injury Distribution")
        injury_counts = df['Injury_Indicator'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(injury_counts, labels=injury_counts.index, autopct='%1.1f%%', startangle=90, colors=['skyblue', 'salmon'])
        ax.axis('equal')
        st.pyplot(fig)

    # Correlation heatmap
    st.write("### Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Histogram of numeric features
    st.write("### Numeric Feature Distributions")
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

    # Input new data for prediction
    st.sidebar.header("Predict Injury Risk")
    st.sidebar.write("Input new data to predict injury risk.")
    top_features = ['ACL_Risk_Score', 'Load_Balance_Score', 'Training_Intensity', 'Training_Load',
                     'Recovery_Days_Per_Week', 'Fatigue_Score', 'Training_Hours_Per_Week']

    input_data = {}
    for feature in top_features:
        value = st.sidebar.number_input(f"{feature}", value=0.0)
        input_data[feature] = value

    if st.sidebar.button("Predict Injury Risk"):
        # Create dataframe for new input
        input_df = pd.DataFrame([input_data])


        # Prediction
        prediction = model.predict(input_df)
        prediction_proba = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")
        injury_label = "Injured" if prediction[0] == 1 else "Not Injured"
        st.write(f"**Predicted Injury Status:** {injury_label}")
        st.write(f"**Injury Probability:** {prediction_proba:.2%}")

else:
    st.info("Awaiting CSV file upload to display data and enable predictions.")

# Footer
st.markdown("---")
st.markdown("**Developed by Ayomitan Adesua and Momoreoluwa Akinwande!**")

