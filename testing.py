import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle

# Load the saved model
model_path = "../Data-Science-Capstone-Project/models/Random Forest.pkl"
loaded_model = pickle.load(open(model_path, "rb"))

category_col = ['Brand', 'Model', 'Variant', 'Fuel', 'Seller_Type', 'Transmission', 'Owner']

# Streamlit app
def preprocess_data(df, selected_features):
    # Use the same LabelEncoder as in the training code
    label_encoder = LabelEncoder()

    for feature in selected_features:
        if feature in category_col:
            df[feature] = label_encoder.fit_transform(df[feature])

    # Drop the target variable
    df = df.drop(["Selling_Price"], axis=1)

    return df

def main():
    st.title("Car Selling Price Prediction")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        st.subheader("Preview of Uploaded Data")
        st.dataframe(df.head())

        st.subheader("Columns in Uploaded Data")
        st.write(df.columns)

        # Allow user to select features for prediction
        selected_features = st.multiselect("Select features for prediction", df.columns)

        # Preprocess the data based on selected features
        df_processed = preprocess_data(df.copy(), selected_features)

        st.subheader("Processed Data Ready for Prediction")
        st.dataframe(df_processed.head())

        # Make predictions
        predictions = loaded_model.predict(df_processed)

        st.subheader("Predicted Selling Prices")
        st.write(predictions)

if __name__ == "__main__":
    main()
