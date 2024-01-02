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

def preprocess_data(df, label_encoders):
    for feature in df.columns:
        if feature in label_encoders:
            df[feature] = label_encoders[feature].transform(df[feature])
    return df

# Display the columns in the web app
st.title("Car Selling Price Prediction App")

# Display loaded CSV data
st.subheader("Loaded CSV Data:")
st.write(cleaned_data)

# Load the LabelEncoders used during training
label_encoders = {}
for feature in category_col:
    label_encoder = LabelEncoder()
    label_encoder.fit(cleaned_data[feature])
    label_encoders[feature] = label_encoder

# Display encoded data
st.subheader("Encoded Data:")
encoded_data = preprocess_data(cleaned_data.copy(), label_encoders)
st.write(encoded_data)

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
    'Brand': [selected_brand],
    'Model': [selected_model],
    'Variant': [selected_variant],
    'Year': [year],
    'Km_Driven': [km_driven],
    'Fuel': [selected_fuel],
    'Seller_Type': [selected_seller_type],
    'Transmission': [selected_transmission],
    'Owner': [selected_owner]
})

# Preprocess the input data
input_data_processed = preprocess_data(input_data.copy(), label_encoders)

# Check if the loaded model and input data are correct
st.subheader("Loaded Model:")
st.write(loaded_model)

st.subheader("Processed Input Data:")
st.write(input_data_processed)

# Initialize the StandardScaler and fit only on numerical features
scaler = StandardScaler()
numerical_features = ['Year', 'Km_Driven']
scaled_data = scaler.fit_transform(cleaned_data[numerical_features])

# Reshape the input data for scaling
input_data_numerical = input_data_processed[numerical_features]
input_data_scaled = scaler.transform(input_data_numerical)

# Display the reshaped input data
st.subheader("Processed Input Data (After Reshaping):")
st.write(input_data_scaled)

# Display the LabelEncoders used during training
st.subheader("Label Encoders:")
st.write(label_encoders)

# Display the StandardScaler used during training
st.subheader("Standard Scaler:")
st.write(scaler)

# Display the columns used during training
st.subheader("Columns used during Training:")
st.write(category_col + numerical_features)

# Button to trigger prediction
if st.button("Predict Selling Price"):
    # Combine numerical and categorical features for prediction
    input_data_combined = pd.concat([input_data_processed[category_col], pd.DataFrame(input_data_scaled, columns=numerical_features)], axis=1)
    # Make predictions
    prediction = loaded_model.predict(input_data_combined)
    # Display the predicted selling price
    st.subheader("Predicted Selling Price:")
    st.write(prediction[0])
