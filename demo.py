import pandas as pd
import pickle
import streamlit as st
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model_path = "../Data-Science-Capstone-Project/models/Random Forest.pkl"
load_model = pickle.load(open(model_path, "rb"))

# Load data
data = pd.read_csv('../Data-Science-Capstone-Project/data/processed/Processed CAR DETAILS.csv')

# Drop the target variable
data = data.drop("Selling_Price", axis=1)

# Label encode categorical columns
label_encoder = LabelEncoder()
for feature in data.select_dtypes(include="object").columns:
    data[feature] = label_encoder.fit_transform(data[feature])

# Streamlit app
st.title('Car Selling Price Prediction')

# Input form for user to input features
year = st.slider('Year', min_value=2000, max_value=2022)
km_driven = st.number_input('Km Driven')
brand = st.selectbox('Brand', data['Brand'].unique())
model_name = st.selectbox('Model', data['Model'].unique())
variant = st.selectbox('Variant', data['Variant'].unique())
fuel = st.selectbox('Fuel', data['Fuel'].unique())
seller_type = st.selectbox('Seller Type', data['Seller_Type'].unique())
transmission = st.selectbox('Transmission', data['Transmission'].unique())
owner = st.selectbox('Owner', data['Owner'].unique())

# Encode categorical input for prediction
input_data = pd.DataFrame({
    'Year': [year],
    'Km_Driven': [km_driven],
    'Brand': [label_encoder.transform([brand])[0]],
    'Model': [label_encoder.transform([model_name])[0]],
    'Variant': [label_encoder.transform([variant])[0]],
    'Fuel': [label_encoder.transform([fuel])[0]],
    'Seller_Type': [label_encoder.transform([seller_type])[0]],
    'Transmission': [label_encoder.transform([transmission])[0]],
    'Owner': [label_encoder.transform([owner])[0]]
})

# Make prediction using the loaded model
prediction = load_model.predict(input_data)[0]

# Display the prediction
st.subheader(f'Predicted Selling Price: {prediction:.2f}')



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
def preprocess_data(df, label_encoders, scaler, numerical_features):
    for feature in df.columns:
        if feature in label_encoders:
            df[feature] = label_encoders[feature].transform(df[feature])
    # Standardize numerical features
    df_scaled = scaler.transform(df[numerical_features])
    return df_scaled

# Display the columns in the web app
st.title("Car Selling Price Prediction App")

# Display loaded CSV data
st.subheader("Loaded CSV Data:")
st.write(cleaned_data)

# Load the LabelEncoders and StandardScaler used during training
label_encoders = {}
for feature in category_col:
    label_encoder = LabelEncoder()
    label_encoder.fit(cleaned_data[feature])
    label_encoders[feature] = label_encoder

scaler = StandardScaler()
numerical_features = ['Km_Driven', 'Year']  # Add other numerical features if present
scaler.fit(cleaned_data[numerical_features])

# Display encoded data
st.subheader("Encoded Data:")
encoded_data = preprocess_data(cleaned_data.copy(), label_encoders, scaler, numerical_features)
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
    'Km_Driven': [km_driven],
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
input_data_processed = preprocess_data(input_data.copy(), label_encoders, scaler, numerical_features)

# Check if the loaded model and input data are correct
st.subheader("Loaded Model:")
st.write(loaded_model)

st.subheader("Processed Input Data:")
st.write(input_data_processed)

# Button to trigger prediction
if st.button("Predict Selling Price"):
    # Make predictions
    prediction = loaded_model.predict(input_data_processed)

    # Display the predicted selling price
    st.subheader("Predicted Selling Price:")
    st.write(prediction)
