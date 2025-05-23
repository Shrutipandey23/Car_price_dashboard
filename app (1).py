import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

st.title("Car Price Prediction Dashboard")

# Upload CSV file
data_file = st.file_uploader("Upload the Car Dataset CSV", type=["csv"])

if data_file is not None:
    df = pd.read_csv(data_file)

    st.subheader("Raw Dataset")
    st.write(df.head())

    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())

    # Replace placeholders with NaN and drop duplicates
    placeholders = ["**", "NA", "Not Available"]
    df.replace(placeholders, np.nan, inplace=True)
    df.drop_duplicates(inplace=True)

    st.subheader("Missing Values")
    st.write(df.isna().sum())

    st.subheader("Summary Statistics")
    st.write(df.describe())

    if "Price" in df.columns:
        st.subheader("Price Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['Price'], kde=True, ax=ax)
        ax.set_title("Distribution of Car Prices")
        st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    if 'Brand' in df.columns and 'Price' in df.columns:
        st.subheader("Price by Brand")
        fig, ax = plt.subplots(figsize=(14,6))
        sns.boxplot(data=df, x='Brand', y='Price', ax=ax)
        ax.set_title("Car Price by Brand")
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Encode categorical columns
    le_dict = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        le_dict[col] = le  # Save encoders for later use

    if 'Price' in df.columns:
        X = df.drop('Price', axis=1)
        y = df['Price']

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("Model Evaluation")
        st.write("R² Score:", r2_score(y_test, y_pred))
        st.write("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))

        if st.checkbox("Predict Car Price"):
            st.subheader("Input Features for Prediction")

            input_vals = []
            for col in X.columns:
                if col in le_dict:
                    options = list(le_dict[col].classes_)
                    val = st.selectbox(f"{col}", options)
                    encoded_val = le_dict[col].transform([val])[0]
                    input_vals.append(encoded_val)
                else:
                    val = st.number_input(f"{col}", value=0.0)
                    input_vals.append(val)

            input_array = scaler.transform([input_vals])
            prediction = model.predict(input_array)
            st.success(f"Predicted Price: ${prediction[0]:,.2f}")
    else:
        st.warning("The dataset does not contain a 'Price' column for modeling.")
else:
    st.info("Please upload a CSV file to begin.")

    
       
