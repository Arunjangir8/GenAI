import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import os

st.set_page_config(page_title="House Price Predictor", layout="wide")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'encoders' not in st.session_state:
    st.session_state.encoders = {}
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = []

def load_and_preprocess_data():
    """Load and preprocess the housing data"""
    df = pd.read_csv('Dataset/Raw/Indian_housing_Delhi_data.csv')
    
    # Extract numeric value from house_size
    df['size_sqft'] = df['house_size'].str.extract('(\d+)').astype(float)
    
    # Extract BHK from house_type
    df['bhk'] = df['house_type'].str.extract('(\d+)').fillna(1).astype(int)
    
    # Fill missing values
    df['numBathrooms'].fillna(df['numBathrooms'].median(), inplace=True)
    df['numBalconies'].fillna(0, inplace=True)
    
    # Select features
    feature_cols = ['bhk', 'size_sqft', 'latitude', 'longitude', 'numBathrooms', 
                    'numBalconies', 'location', 'Status']
    
    df_clean = df[feature_cols + ['price']].dropna()
    
    return df_clean, feature_cols

def train_model(df, feature_cols):
    """Train the model"""
    encoders = {}
    df_encoded = df.copy()
    
    # Encode categorical features
    for col in ['location', 'Status']:
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

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Train Model", "Make Prediction"])

# Main content
if page == "Train Model":
    st.title("🏠 House Price Prediction - Model Training")
    
    if st.button("Load Data & Train Model", type="primary"):
        with st.spinner("Loading data and training model..."):
            try:
                df, feature_cols = load_and_preprocess_data()
                st.success(f"✅ Data loaded: {len(df)} records")
                
                model, encoders, mae, r2, df_full = train_model(df, feature_cols)
                
                st.session_state.model = model
                st.session_state.encoders = encoders
                st.session_state.feature_cols = feature_cols
                st.session_state.df = df_full
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Absolute Error", f"₹{mae:,.0f}")
                with col2:
                    st.metric("R² Score", f"{r2:.3f}")
                
                st.success("✅ Model trained successfully!")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    if st.session_state.model is not None:
        st.info("Model is ready for predictions! Go to 'Make Prediction' page.")

elif page == "Make Prediction":
    st.title("🔮 House Price Prediction")
    
    if st.session_state.model is None:
        st.warning("⚠️ Please train the model first!")
    else:
        st.success("✅ Model loaded and ready!")
        
        col1, col2 = st.columns(2)
        
        with col1:
            bhk = st.selectbox("BHK", [1, 2, 3, 4, 5, 6])
            size_sqft = st.number_input("Size (sq ft)", min_value=100, max_value=10000, value=1000)
            num_bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
            num_balconies = st.number_input("Balconies", min_value=0, max_value=5, value=1)
        
        with col2:
            df = st.session_state.df
            location = st.selectbox("Location", sorted(df['location'].unique()))
            status = st.selectbox("Furnishing Status", sorted(df['Status'].unique()))
            latitude = st.number_input("Latitude", value=28.6139, format="%.4f")
            longitude = st.number_input("Longitude", value=77.2090, format="%.4f")
        
        if st.button("Predict Price", type="primary"):
            try:
                input_data = pd.DataFrame({
                    'bhk': [bhk],
                    'size_sqft': [size_sqft],
                    'latitude': [latitude],
                    'longitude': [longitude],
                    'numBathrooms': [num_bathrooms],
                    'numBalconies': [num_balconies],
                    'location': [st.session_state.encoders['location'].transform([location])[0]],
                    'Status': [st.session_state.encoders['Status'].transform([status])[0]]
                })
                
                prediction = st.session_state.model.predict(input_data)[0]
                
                st.markdown("---")
                st.subheader("Predicted Price")
                st.markdown(f"<h1 style='text-align: center; color: #4CAF50;'>₹{prediction:,.0f}</h1>", 
                           unsafe_allow_html=True)
                st.markdown(f"<h3 style='text-align: center;'>₹{prediction/12:,.0f} per month</h3>", 
                           unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
