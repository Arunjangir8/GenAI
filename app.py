import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

st.set_page_config(page_title="House Price Predictor", layout="wide")

def load_and_preprocess_data():
    """Load and preprocess the housing data from all cities"""
    # Load all three datasets
    df1 = pd.read_csv('Dataset/Raw/Indian_housing_Delhi_data.csv')
    df2 = pd.read_csv('Dataset/Raw/Indian_housing_Mumbai_data.csv')
    df3 = pd.read_csv('Dataset/Raw/Indian_housing_Pune_data.csv')
    
    # Combine all datasets
    df = pd.concat([df1, df2, df3], ignore_index=True)
    
    # Extract numeric value from house_size
    df['size_sqft'] = df['house_size'].str.replace(',', '').str.extract('(\d+)').astype(float)
    
    # Extract BHK from house_type
    df['bhk'] = df['house_type'].str.extract('(\d+)').fillna(1).astype(int)
    
    # Fill missing values
    df['numBathrooms'].fillna(df['numBathrooms'].median(), inplace=True)
    df['numBalconies'].fillna(0, inplace=True)
    
    # Select features
    feature_cols = ['bhk', 'size_sqft', 'latitude', 'longitude', 'numBathrooms', 
                    'numBalconies', 'city', 'location', 'Status']
    
    df_clean = df[feature_cols + ['price']].dropna()
    
    return df_clean, feature_cols

def train_model(df, feature_cols):
    """Train the model"""
    encoders = {}
    df_encoded = df.copy()
    
    # Encode categorical features
    for col in ['city', 'location', 'Status']:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        encoders[col] = le
    
    X = df_encoded[feature_cols]
    y = df_encoded['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return model, encoders, mae, r2, df

# Auto-train model on startup
@st.cache_resource
def initialize_model():
    with st.spinner("Loading data and training model..."):
        df, feature_cols = load_and_preprocess_data()
        model, encoders, mae, r2, df_full = train_model(df, feature_cols)
        return model, encoders, feature_cols, df_full, mae, r2

model, encoders, feature_cols, df, mae, r2 = initialize_model()

# Main UI
st.title("House Price Prediction")

# Show model metrics in sidebar
with st.sidebar:
    st.subheader("Model Performance")
    st.metric("MAE", f"₹{mae:,.0f}")
    st.metric("R² Score", f"{r2:.3f}")
    st.metric("Training Data", f"{len(df):,} records")

col1, col2 = st.columns(2)

with col1:
    city = st.selectbox("City", sorted(df['city'].unique()))
    bhk = st.selectbox("BHK", [1, 2, 3, 4, 5, 6])
    size_sqft = st.number_input("Size (sq ft)", min_value=100, max_value=10000, value=1000)
    num_bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

with col2:
    locations = sorted(df[df['city'] == city]['location'].unique())
    location = st.selectbox("Location", locations)
    status = st.selectbox("Furnishing Status", sorted(df['Status'].unique()))
    num_balconies = st.number_input("Balconies", min_value=0, max_value=5, value=1)
    
    # Auto-fill lat/long based on city
    city_coords = {'Delhi': (28.6139, 77.2090), 'Mumbai': (19.0760, 72.8777), 'Pune': (18.5204, 73.8567)}
    default_lat, default_lon = city_coords.get(city, (28.6139, 77.2090))
    latitude = st.number_input("Latitude", value=default_lat, format="%.4f")
    longitude = st.number_input("Longitude", value=default_lon, format="%.4f")

if st.button("Predict Price", type="primary", use_container_width=True):
    try:
        input_data = pd.DataFrame({
            'bhk': [bhk],
            'size_sqft': [size_sqft],
            'latitude': [latitude],
            'longitude': [longitude],
            'numBathrooms': [num_bathrooms],
            'numBalconies': [num_balconies],
            'city': [encoders['city'].transform([city])[0]],
            'location': [encoders['location'].transform([location])[0]],
            'Status': [encoders['Status'].transform([status])[0]]
        })
        
        prediction = model.predict(input_data)[0]
        
        st.markdown("---")
        st.subheader("Predicted Price")
        st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>₹{prediction:,.0f}</h1>", 
                   unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align: center;'>₹{prediction/12:,.0f} per month</h3>", 
                   unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
