import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the saved model
model_filename = "../Data-Science-Capstone-Project/models/Random Forest.pkl"
loaded_model = pickle.load(open(model_filename, "rb"))

# Load the cleaned data
cleaned_data_filename = "../Data-Science-Capstone-Project/data/processed/Processed CAR DETAILS.csv"
cleaned_data = pd.read_csv(cleaned_data_filename)

category_col = ['Brand', 'Model', 'Variant', 'Fuel', 'Seller_Type', 'Transmission', 'Owner']

# Streamlit app
def preprocess_data(df, selected_features):
    # Use the same LabelEncoder as in the training code
    label_encoder = LabelEncoder()

    for feature in selected_features:
        if feature in category_col:
            df[feature] = label_encoder.fit_transform(df[feature])

    # Drop the target variable
    #df = df.drop(["Selling_Price"], axis=1)

    return df

# Display the columns in the web app
st.title("Car Selling Price Prediction App")

# Display sliders for numerical features
km_driven = st.slider("Select KM Driven:", min_value=cleaned_data["Km_Driven"].min(), max_value=cleaned_data["Km_Driven"].max())
year = st.slider("Select Year:", min_value=cleaned_data["Year"].min(), max_value=cleaned_data["Year"].max())

# Display dropdowns for categorical features
selected_brand = st.selectbox("Select Brand:", cleaned_data["Brand"].unique())
selected_model = st.selectbox("Select Model:", cleaned_data["Model"].unique())
selected_variant = st.selectbox("Select Variant:", cleaned_data["Variant"].unique())
selected_fuel = st.selectbox("Select Fuel:", cleaned_data["Fuel"].unique())
selected_seller_type = st.selectbox("Select Seller Type:", cleaned_data["Seller_Type"].unique())
selected_transmission = st.selectbox("Select Transmission:", cleaned_data["Transmission"].unique())
selected_owner = st.selectbox("Select Owner:", cleaned_data["Owner"].unique())

# Create a DataFrame from the user inputs
input_data = pd.DataFrame({
    'Kms_Driven': [km_driven],
    'Year': [year],
    'Brand': [selected_brand],
    'Model': [selected_model],
    'Variant': [selected_variant],
    'Fuel': [selected_fuel],
    'Seller_Type': [selected_seller_type],
    'Transmission': [selected_transmission],
    'Owner': [selected_owner]
})

# Preprocess the input data
input_data_processed = preprocess_data(input_data.copy(), category_col)

# Standardize the input data
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data_processed)

# Button to trigger prediction
if st.button("Predict Selling Price"):
    # Make predictions
    prediction = loaded_model.predict(input_data_scaled)

    # Display the predicted selling price
    st.subheader("Predicted Selling Price:")
    st.write(prediction)
