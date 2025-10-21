"""
Metro Manila Class/Work Suspension Prediction System
A Streamlit application for predicting and analyzing suspension declarations
HERALD v2.0 - Enhanced with PAGASA criteria and improved UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio 
from plotly.subplots import make_subplots
from fpdf import FPDF
import joblib
import json
import requests
from datetime import datetime, timedelta
import tempfile 
import os
from typing import Dict, Tuple, Optional, List
import kaleido
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="HERALD v2.0",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONSTANTS & CONFIGURATIONS
# ============================================================================


CITY_COORDINATES = {
    'Manila': {'lat': 14.5995, 'lon': 120.9842},
    'Quezon City': {'lat': 14.6760, 'lon': 121.0437},
    'Caloocan': {'lat': 14.6488, 'lon': 120.9830},
    'Las Piñas': {'lat': 14.4453, 'lon': 120.9842},
    'Makati': {'lat': 14.5547, 'lon': 121.0244},
    'Malabon': {'lat': 14.6630, 'lon': 120.9573},
    'Mandaluyong': {'lat': 14.5794, 'lon': 121.0359},
    'Marikina': {'lat': 14.6507, 'lon': 121.1029},
    'Muntinlupa': {'lat': 14.4086, 'lon': 121.0390},
    'Navotas': {'lat': 14.6685, 'lon': 120.9409},
    'Parañaque': {'lat': 14.4793, 'lon': 121.0198},
    'Pasay': {'lat': 14.5378, 'lon': 121.0014},
    'Pasig': {'lat': 14.5764, 'lon': 121.0851},
    'Pateros': {'lat': 14.5438, 'lon': 121.0685},
    'San Juan': {'lat': 14.6019, 'lon': 121.0355},
    'Taguig': {'lat': 14.5176, 'lon': 121.0509},
    'Valenzuela': {'lat': 14.7006, 'lon': 120.9830}
}

SUSPENSION_LEVELS = {
    0: {"name": "No Suspension", "color": "#10b981", "emoji": "✅", "risk": "None"},
    1: {"name": "Preschool Only", "color": "#fbbf24", "emoji": "⚠️", "risk": "Low"},
    2: {"name": "Preschool + Elementary", "color": "#f97316", "emoji": "🟠", "risk": "Medium"},
    3: {"name": "All Levels (Public)", "color": "#ef4444", "emoji": "🔴", "risk": "High"},
    4: {"name": "All Levels + Work", "color": "#991b1b", "emoji": "🚨", "risk": "Critical"},
}

# PAGASA Suspension Criteria
PAGASA_CRITERIA = {
    'tcws': [
        {'signal': 1, 'wind_min': 30, 'wind_max': 60, 'suspension': 1, 'description': 'TCWS 1: Winds 30-60 km/h'},
        {'signal': 2, 'wind_min': 60, 'wind_max': 100, 'suspension': 2, 'description': 'TCWS 2: Winds 60-100 km/h'},
        {'signal': 3, 'wind_min': 100, 'wind_max': 150, 'suspension': 3, 'description': 'TCWS 3: Winds 100-150 km/h'},
        {'signal': 4, 'wind_min': 150, 'wind_max': 300, 'suspension': 4, 'description': 'TCWS 4: Winds 150-250 km/h'},
    ],
    'rainfall': [
        {'level': 1, 'precip': 7.5, 'suspension': 1, 'description': 'Light Rainfall Warning: ≥7.5 mm/hour'},
        {'level': 2, 'precip': 15, 'suspension': 3, 'description': 'Moderate Rainfall Warning: ≥15 mm/hour'},
        {'level': 3, 'precip': 30, 'suspension': 4, 'description': 'Heavy Rainfall Warning: ≥30 mm/hour'},
    ],
    'heat_index': [
        {'level': 1, 'temp': 41, 'suspension': 3, 'description': 'Heat Index 41°C: Outdoor activities suspended'},
        {'level': 2, 'temp': 51, 'suspension': 4, 'description': 'Heat Index 51°C: All outdoor work suspended'},
    ]
}

# ============================================================================
# CACHING & LOADING FUNCTIONS
# ============================================================================

@st.cache_resource(show_spinner=False)
def load_model_artifacts():
    """Load all pre-trained model artifacts at app startup"""
    try:
        artifacts = {
            'model': joblib.load('model.pkl'),
            'preprocessor': joblib.load('preprocessor.pkl'),
            'city_encoder': joblib.load('city_encoder.pkl'),
            'label_encoder': joblib.load('label_encoder.pkl')
        }
        
        with open('metrics.json', 'r') as f:
            artifacts['metrics'] = json.load(f)
        with open('feature_importance.json', 'r') as f:
            artifacts['feature_importance'] = json.load(f)
        with open('thresholds.json', 'r') as f:
            artifacts['thresholds'] = json.load(f)
            
        return artifacts
    except FileNotFoundError as e:
        st.error(f"Model artifacts not found: {str(e)}")
        st.info("Please run 'python train_model.py' first.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        st.stop()

@st.cache_data(show_spinner=False)
def load_historical_data():
    """Load and cache the entire dataset"""
    try:
        df = pd.read_csv('metro_manila_weather_sus_data.csv', parse_dates=['date'])
        return df
    except FileNotFoundError:
        st.error("Dataset not found! Please ensure 'metro_manila_weather_sus_data.csv' exists.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        st.stop()

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_date_bounds():
    """Get the date bounds for date pickers"""
    min_date = datetime(2020, 9, 20).date()
    max_date = datetime.now().date() + timedelta(days=7)
    return min_date, max_date

def get_data_mode(selected_date: datetime, dataset_start: datetime, dataset_end: datetime) -> str:
    if dataset_start.date() <= selected_date.date() <= dataset_end.date():
        return 'historical'
    else:
        return 'forecast'

def get_historical_weather(df: pd.DataFrame, date: datetime, city: str) -> Dict:
    mask = (df['date'].dt.date == date.date()) & (df['city'] == city)
    city_data = df[mask].copy()
    
    if len(city_data) == 0:
        return None
    
    hourly_data = city_data.sort_values('date')
    
    return {
        'mode': 'historical',
        'date': date,
        'city': city,
        'hourly_data': hourly_data,
        'weather_data': {
            'temperature_2m': hourly_data['temperature_2m'].mean(),
            'relativehumidity_2m': hourly_data['relativehumidity_2m'].mean(),
            'precipitation': hourly_data['precipitation'].sum(),
            'apparent_temperature': hourly_data['apparent_temperature'].mean(),
            'windspeed_10m': hourly_data['windspeed_10m'].mean()
        },
        'suspension_level': int(hourly_data['suspension'].mode()[0]),
        'is_prediction': False
    }

@st.cache_data(ttl=3600)
def fetch_openmeteo_forecast(date: datetime, city: str) -> Optional[Dict]:
    coords = CITY_COORDINATES.get(city)
    if not coords:
        return None
    
    try:
        date_str = date.strftime('%Y-%m-%d')
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': coords['lat'],
            'longitude': coords['lon'],
            'hourly': 'temperature_2m,relativehumidity_2m,precipitation,apparent_temperature,windspeed_10m',
            'start_date': date_str,
            'end_date': date_str,
            'timezone': 'Asia/Manila'
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        hourly = data.get('hourly', {})
        times = pd.to_datetime(hourly['time'])
        
        forecast_df = pd.DataFrame({
            'date': times,
            'temperature_2m': hourly['temperature_2m'],
            'relativehumidity_2m': hourly['relativehumidity_2m'],
            'precipitation': hourly['precipitation'],
            'apparent_temperature': hourly['apparent_temperature'],
            'windspeed_10m': hourly['windspeed_10m']
        })
        
        return {
            'mode': 'forecast',
            'date': date,
            'city': city,
            'hourly_data': forecast_df,
            'weather_data': {
                'temperature_2m': forecast_df['temperature_2m'].mean(),
                'relativehumidity_2m': forecast_df['relativehumidity_2m'].mean(),
                'precipitation': forecast_df['precipitation'].sum(),
                'apparent_temperature': forecast_df['apparent_temperature'].mean(),
                'windspeed_10m': forecast_df['windspeed_10m'].mean()
            }
        }
    except Exception as e:
        st.error(f"Failed to fetch weather forecast: {str(e)}")
        return None

def engineer_features_for_prediction(weather_data: Dict, city: str, hourly_df: pd.DataFrame, artifacts: Dict) -> pd.DataFrame:
    if len(hourly_df) >= 12:
        row = hourly_df.iloc[12].copy()
    else:
        row = hourly_df.iloc[0].copy()
    
    features = {
        'relativehumidity_2m': weather_data['relativehumidity_2m'],
        'temperature_2m': weather_data['temperature_2m'],
        'precipitation': weather_data['precipitation'],
        'apparent_temperature': weather_data['apparent_temperature'],
        'windspeed_10m': weather_data['windspeed_10m'],
    }
    
    dt = hourly_df['date'].iloc[0]
    features['hour'] = dt.hour
    features['day_of_week'] = dt.dayofweek
    features['month'] = dt.month
    features['is_weekend'] = 1 if dt.dayofweek >= 5 else 0
    features['is_rush_hour'] = 1 if dt.hour in [7, 8, 9, 17, 18, 19] else 0
    
    for window in [3, 6, 12]:
        end_idx = min(len(hourly_df), window)
        features[f'precip_roll_{window}h'] = hourly_df['precipitation'][:end_idx].mean()
        features[f'wind_roll_{window}h'] = hourly_df['windspeed_10m'][:end_idx].mean()
    
    for lag in [1, 2, 3]:
        if lag < len(hourly_df):
            features[f'precip_lag_{lag}h'] = hourly_df['precipitation'].iloc[-lag]
            features[f'temp_lag_{lag}h'] = hourly_df['temperature_2m'].iloc[-lag]
        else:
            features[f'precip_lag_{lag}h'] = 0
            features[f'temp_lag_{lag}h'] = weather_data['temperature_2m']
    
    features['is_precip_peak'] = 1 if features['precipitation'] > features['precip_roll_6h'] * 1.5 else 0
    features['is_wind_peak'] = 1 if features['windspeed_10m'] > features['wind_roll_6h'] * 1.3 else 0
    features['apparent_temp_delta'] = features['apparent_temperature'] - features['temperature_2m']
    
    precip = features['precipitation']
    if precip <= 0:
        features['precip_intensity'] = 0
    elif precip <= 5:
        features['precip_intensity'] = 1
    elif precip <= 15:
        features['precip_intensity'] = 2
    elif precip <= 50:
        features['precip_intensity'] = 3
    else:
        features['precip_intensity'] = 4
    
    wind = features['windspeed_10m']
    if wind <= 20:
        features['wind_category'] = 0
    elif wind <= 40:
        features['wind_category'] = 1
    elif wind <= 60:
        features['wind_category'] = 2
    else:
        features['wind_category'] = 3
    
    city_encoder = artifacts['city_encoder']
    features['city_encoded'] = city_encoder.transform([city])[0]
    
    feature_df = pd.DataFrame([features])
    numerical_cols = ['relativehumidity_2m', 'temperature_2m', 'precipitation', 'apparent_temperature', 'windspeed_10m', 'apparent_temp_delta']
    feature_df[numerical_cols] = artifacts['preprocessor'].transform(feature_df[numerical_cols])
    
    return feature_df

def predict_suspension(weather_data: Dict, city: str, hourly_df: pd.DataFrame, artifacts: Dict) -> Dict:
    features = engineer_features_for_prediction(weather_data, city, hourly_df, artifacts)
    features = features.astype(float)
    
    model = artifacts['model']
    label_encoder = artifacts['label_encoder']
    
    prediction_encoded = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    prediction = label_encoder.inverse_transform([prediction_encoded])[0]
    confidence = float(probabilities.max())
    
    risk_levels = ['None', 'Low', 'Medium', 'High', 'Critical']
    risk_idx = min(int(prediction), len(risk_levels) - 1)
    risk_level = risk_levels[risk_idx]
    
    return {
        'predicted_level': int(prediction),
        'probabilities': probabilities.tolist(),
        'confidence': confidence,
        'risk_level': risk_level,
        'is_prediction': True
    }

def check_pagasa_criteria(weather_data: Dict) -> Tuple[int, List[str]]:
    """Check PAGASA suspension criteria and return suspension level and reasons"""
    suspension_level = 0
    reasons = []
    
    wind = weather_data['windspeed_10m']
    for criterion in PAGASA_CRITERIA['tcws']:
        if criterion['wind_min'] <= wind <= criterion['wind_max']:
            suspension_level = max(suspension_level, criterion['suspension'])
            reasons.append(f"{criterion['description']}")
            break
    
    precip = weather_data['precipitation']
    for criterion in PAGASA_CRITERIA['rainfall']:
        if precip >= criterion['precip']:
            suspension_level = max(suspension_level, criterion['suspension'])
            reasons.append(f"{criterion['description']}")
    
    heat_index = weather_data['apparent_temperature']
    for criterion in PAGASA_CRITERIA['heat_index']:
        if heat_index >= criterion['temp']:
            suspension_level = max(suspension_level, criterion['suspension'])
            reasons.append(f"{criterion['description']}")
    
    return suspension_level, reasons

def get_weather_and_suspension(selected_date: datetime, city: str, df: pd.DataFrame, artifacts: Dict) -> Dict:
    mode = get_data_mode(selected_date, df['date'].min(), df['date'].max())
    
    if mode == 'historical':
        return get_historical_weather(df, selected_date, city)
    else:
        forecast_data = fetch_openmeteo_forecast(selected_date, city)
        if forecast_data:
            prediction = predict_suspension(forecast_data['weather_data'], city, forecast_data['hourly_data'], artifacts)
            return {**forecast_data, **prediction}
        return None

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_suspension_gauge(level: int, is_prediction: bool = False, confidence: float = None):
    info = SUSPENSION_LEVELS[level]
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=level,
        title={'text': f"{'Predicted' if is_prediction else 'Actual'} Suspension Level<br><span style='font-size:0.8em'>{info['name']}</span>"},
        domain={'x': [0, 1], 'y': [0, 1]},
        gauge={
            'axis': {'range': [0, 4], 'tickwidth': 1},
            'bar': {'color': info['color']},
            'steps': [
                {'range': [0, 1], 'color': '#10b981'},
                {'range': [1, 2], 'color': '#fbbf24'},
                {'range': [2, 3], 'color': '#f97316'},
                {'range': [3, 4], 'color': '#ef4444'}
            ],
            'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': level}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=80, b=20), transition={'duration': 800, 'easing': 'cubic-in-out'})
    return fig

def create_hourly_weather_chart(hourly_df: pd.DataFrame, mode: str):
    fig = make_subplots(rows=3, cols=1, subplot_titles=('Temperature (°C)', 'Precipitation (mm)', 'Wind Speed (km/h)'), vertical_spacing=0.1)
    
    fig.add_trace(go.Scatter(x=hourly_df['date'], y=hourly_df['temperature_2m'], name='Temperature', line=dict(color='#ef4444', width=2), fill='tozeroy', fillcolor='rgba(239, 68, 68, 0.1)'), row=1, col=1)
    fig.add_trace(go.Bar(x=hourly_df['date'], y=hourly_df['precipitation'], name='Precipitation', marker_color='#3b82f6'), row=2, col=1)
    fig.add_trace(go.Scatter(x=hourly_df['date'], y=hourly_df['windspeed_10m'], name='Wind Speed', line=dict(color='#10b981', width=2)), row=3, col=1)
    
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_layout(height=600, showlegend=False, margin=dict(l=50, r=20, t=80, b=50))
    return fig


def create_weather_cards(weather_data: Dict):
    cols = st.columns(5)
    metrics = [
        ("Temperature", f"{weather_data['temperature_2m']:.1f}°C"),
        ("Humidity", f"{weather_data['relativehumidity_2m']:.1f}%"),
        ("Precipitation", f"{weather_data['precipitation']:.1f}mm"),
        ("Feels Like", f"{weather_data['apparent_temperature']:.1f}°C"),
        ("Wind Speed", f"{weather_data['windspeed_10m']:.1f}km/h")
    ]
    for col, (label, value) in zip(cols, metrics):
        with col:
            st.metric(label, value)

def fetch_city_suspension_data(city_name: str, selected_date: datetime, df: pd.DataFrame, artifacts: Dict) -> Optional[Dict]:
    """
    Thread-safe function to fetch suspension data for a single city.
    Returns city data dictionary or None if fetch fails.
    """
    try:
        city_data = get_weather_and_suspension(
            datetime.combine(selected_date, datetime.min.time()), 
            city_name, 
            df, 
            artifacts
        )
        
        if city_data:
            level = city_data.get('predicted_level', city_data.get('suspension_level', 0))
            coords = CITY_COORDINATES[city_name]
            return {
                'city': city_name,
                'lat': coords['lat'],
                'lon': coords['lon'],
                'suspension_level': level,
                'color': SUSPENSION_LEVELS[level]['color'],
                'name': SUSPENSION_LEVELS[level]['name']
            }
        return None
    except Exception as e:
        st.warning(f"Failed to fetch data for {city_name}: {str(e)}")
        return None
    
def fetch_all_cities_parallel(selected_date: datetime, df: pd.DataFrame, artifacts: Dict, max_workers: int = 5) -> pd.DataFrame:
    """
    Fetch suspension data for all cities in parallel using ThreadPoolExecutor.
    
    Args:
        selected_date: Date to fetch data for
        df: Historical data DataFrame
        artifacts: Model artifacts
        max_workers: Maximum number of concurrent threads (default: 5)
    
    Returns:
        DataFrame with all cities' suspension data
    """
    all_cities_data = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all city fetches concurrently
        future_to_city = {
            executor.submit(fetch_city_suspension_data, city_name, selected_date, df, artifacts): city_name
            for city_name in CITY_COORDINATES.keys()
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_city):
            city_name = future_to_city[future]
            try:
                city_data = future.result()
                if city_data:
                    all_cities_data.append(city_data)
            except Exception as e:
                st.warning(f"Error processing {city_name}: {str(e)}")
    
    return pd.DataFrame(all_cities_data)

# ============================================================================
# PAGE: HOME
# ============================================================================

def page_home():
    st.markdown(
    "<h1 style='font-size: 50px;'>🌤️HERALD v2.0</h1>",
    unsafe_allow_html=True
)
    
    df = load_historical_data()
    artifacts = load_model_artifacts()
    min_date, max_date = get_date_bounds()
    
    with st.sidebar:
        st.markdown("### Selection Settings")
        selected_date = st.date_input(
            "Select Date", 
            value=datetime.now().date(), 
            min_value=min_date, 
            max_value=max_date, 
            key='home_date_input'
        )
        
        city = st.selectbox("Select City", sorted(CITY_COORDINATES.keys()), 
                           index=sorted(CITY_COORDINATES.keys()).index('Manila'), 
                           key='home_city')
        
        mode = get_data_mode(datetime.combine(selected_date, datetime.min.time()), 
                            df['date'].min(), df['date'].max())
        if mode == 'historical':
            st.success("Historical Mode - Using Actual Data")
        else:
            st.info("Forecast Mode - Using Open-Meteo API")
        
        if st.button("Change to Today", use_container_width=True):
            selected_date = datetime.now().date()
            st.rerun()
    
    
    st.markdown(f"<div style='padding: 15px; background-color: #009cdf; border-radius: 10px; margin-bottom: 20px;'><h3 style='margin: 0; color: #FFFFFF;font-size: 40px'> 🗺️ {city} |🧭 {selected_date.strftime('%B %d, %Y')}</h3></div>", unsafe_allow_html=True)
    
    data = get_weather_and_suspension(datetime.combine(selected_date, datetime.min.time()), city, df, artifacts)
    
    if not data:
        st.error("No data available for selected date and city.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        level = data.get('predicted_level', data.get('suspension_level', 0))
        is_pred = data.get('is_prediction', False)
        conf = data.get('confidence')
        fig = create_suspension_gauge(level, is_pred, conf)
        st.plotly_chart(fig, use_container_width=True)
        if is_pred:

            st.metric("Confidence", f"{conf*100:.1f}%")
            st.caption("ML Prediction")

        else:
            st.caption("Historical Record")
    
    with col2:
        info = SUSPENSION_LEVELS[level]
        st.markdown(f"<div style='padding: 20px; background-color: {info['color']}20; border-left: 4px solid {info['color']}; border-radius: 5px;'><h2 style='margin:0; font-size: 40px;color: {info['color']};'>{info['emoji']} {info['name']}</h2><p style='margin: 10px 0 0 0; font-size:27px;'><strong>Risk Level:</strong> {info['risk']}</p></div>", unsafe_allow_html=True)
        st.markdown("""<h3 style="font-size: 32px; color: #000103;">What This Means</h3> """, unsafe_allow_html=True)
        descriptions = {0: "Normal school and work operations. Weather conditions are safe.", 1: "Only preschool classes are suspended due to weather conditions.", 2: "Preschool and elementary classes are suspended.", 3: "All school levels (public schools) are suspended.", 4: "All school levels and government work are suspended."}
        st.info(descriptions[level])
    
    st.markdown("---")
    st.markdown(f"""<h3 style="font-size: 40px; color: #000103;"> Weather Metrics for {city} | {selected_date.strftime('%B %d, %Y')}</h3>""", unsafe_allow_html=True)

    
    create_weather_cards(data['weather_data'])
    
    st.markdown("---")
    st.markdown(f"""<h3 style= "font-size:40px;">Metro Manila Suspension Map </h3>""", unsafe_allow_html=True)

    # Show loading indicator
    with st.spinner('Loading suspension data for all cities...'):
        # Fetch all cities data in parallel
        map_df = fetch_all_cities_parallel(selected_date, df, artifacts, max_workers=8)

    if len(map_df) == 0:
        st.warning("Unable to fetch suspension data for cities.")
    else:
        fig = go.Figure()
        
        level_colors = {
            0: '#22C55E',
            1: '#FBBF24',
            2: '#F97316',
            3: '#EF4444',
            4: '#7C2D12',
        }
        
        for level in range(5):
            level_data = map_df[map_df['suspension_level'] == level]
            if len(level_data) > 0:
                fig.add_trace(go.Scattermapbox(
                    lat=level_data['lat'], 
                    lon=level_data['lon'], 
                    mode='markers', 
                    marker=dict(size=20, color=level_colors[level], opacity=0.9),
                    text=level_data['city'],
                    customdata=level_data[['name']], 
                    hovertemplate='<b>%{text}</b><br>Level: %{customdata[0]}<br><extra></extra>', 
                    name=SUSPENSION_LEVELS[level]['name'],
                    legendgroup=f'level{level}'
                ))
        
        selected_city_data = map_df[map_df['city'] == city]
        if len(selected_city_data) > 0:
            fig.add_trace(go.Scattermapbox(
                lat=selected_city_data['lat'], 
                lon=selected_city_data['lon'], 
                mode='markers+text', 
                marker=dict(size=25, color='rgba(0,0,0,0.2)', symbol='star'),
                text='●',
                textfont=dict(size=30, color='black'),
                hoverinfo='skip', 
                showlegend=False
            ))
        
        fig.update_layout(
            mapbox=dict(style='open-street-map', center=dict(lat=14.58, lon=121.0223), zoom=10), 
            height=600, 
            margin=dict(l=0, r=0, t=0, b=0), 
            showlegend=True, 
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="white", bordercolor="black", borderwidth=1, font=dict(size=16))
        )
        st.plotly_chart(fig, use_container_width=True)



# ============================================================================
# PAGE: WEATHER ANALYTICS
# ============================================================================

def page_weather_analytics():
    st.markdown(
    "<h1 style='font-size: 40px;'>🌦️Weather Analytics</h1>",
    unsafe_allow_html=True
)
    
    df = load_historical_data()
    min_date, max_date = get_date_bounds()
    
    with st.sidebar:
        st.markdown("### Selection Settings")
        selected_date = st.date_input(
            "Select Date", 
            value=datetime.now().date(), 
            min_value=min_date, 
            max_value=max_date, 
            key='analytics_date_input'
        )
        
        city = st.selectbox("Select City", sorted(CITY_COORDINATES.keys()), 
                           index=sorted(CITY_COORDINATES.keys()).index('Manila'), 
                           key='analytics_city')
        
        mode = get_data_mode(datetime.combine(selected_date, datetime.min.time()), 
                            df['date'].min(), df['date'].max())
        if mode == 'historical':
            st.success("Historical Mode - Using Actual Data")
        else:
            st.info("Forecast Mode - Using Open-Meteo API")
        
        if st.button("Change to Today", use_container_width=True, key='analytics_today_btn'):
            selected_date = datetime.now().date()
            st.rerun()
    
    data = get_weather_and_suspension(datetime.combine(selected_date, datetime.min.time()), city, df, load_model_artifacts())
    
    if not data:
        st.warning("No data available for the selected date and city.")
        return
    
    st.markdown("---")
    st.markdown(f"""<h2 style  = font-size: 40px;>Daily Summary</h2>""", unsafe_allow_html=True)
    
    
    create_weather_cards(data['weather_data'])
    
    # 8. SUSPENSION LEVEL GAUGE & METRO MANILA SUSPENSION MAP
    # -------------------------------

    col1, col2 = st.columns([1, 2], gap="large")

    with col1:
        level = data.get('predicted_level', data.get('suspension_level', 0))
        is_pred = data.get('is_prediction', False)
        conf = data.get('confidence')
        fig_gauge = create_suspension_gauge(level, is_pred, conf)
        st.plotly_chart(fig_gauge, use_container_width=True)
        if is_pred and conf is not None:
            st.metric("Confidence", f"{conf*100:.1f}%")
            st.caption("ML Prediction")
        else:
            st.caption("Historical Record")

    with col2:
    # --- BUILD THE METRO MANILA SUSPENSION MAP ---
        with st.spinner('Loading map data...'):
            map_df = fetch_all_cities_parallel(selected_date, df, load_model_artifacts(), max_workers=8)
        
        if len(map_df) == 0:
            st.warning("Unable to fetch suspension data for cities.")
        else:
            fig_map = go.Figure()

            level_colors = {
                0: '#22C55E',
                1: '#FBBF24',
                2: '#F97316',
                3: '#EF4444',
                4: '#7C2D12',
            }
            for lvl in range(5):
                level_data = map_df[map_df['suspension_level'] == lvl]
                if len(level_data) > 0:
                    fig_map.add_trace(go.Scattermapbox(
                        lat=level_data['lat'],
                        lon=level_data['lon'],
                        mode='markers',
                        marker=dict(size=20, color=level_colors[lvl], opacity=0.9),
                        text=level_data['city'],
                        customdata=level_data[['name']],
                        hovertemplate='<b>%{text}</b><br>Level: %{customdata[0]}<br><extra></extra>',
                        name=SUSPENSION_LEVELS[lvl]['name'],
                        legendgroup=f'level{lvl}'
                    ))

            selected_city_data = map_df[map_df['city'] == city]
            if len(selected_city_data) > 0:
                fig_map.add_trace(go.Scattermapbox(
                    lat=selected_city_data['lat'],
                    lon=selected_city_data['lon'],
                    mode='markers+text',
                    marker=dict(size=25, color='rgba(0,0,0,0.2)', symbol='star'),
                    text='●',
                    textfont=dict(size=30, color='black'),
                    hoverinfo='skip',
                    showlegend=False
                ))
            fig_map.update_layout(
                mapbox=dict(style='open-street-map',
                            center=dict(lat=14.54, lon=121.0223), zoom=9.5),
                height=400,
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="white", bordercolor="black", borderwidth=1, font=dict(size=16))
            )
            st.plotly_chart(fig_map, use_container_width=True)


    # -------------------------------
    # 9. HOURLY WEATHER TRENDS CHART
    # -------------------------------
    st.markdown("---")
    st.markdown(f"""<h2 style  = font-size: 40px;>Hourly Weather Trends</h2>""", unsafe_allow_html=True)
    

    hourly = data['hourly_data']  # Assumes this is a DataFrame

    fig = go.Figure()

# 1. Temperature - Line trace (First Y axis, left)
    fig.add_trace(go.Scatter(
        x=hourly['date'],
        y=hourly['temperature_2m'],
        name="Temperature (°C)",
        mode='lines',
        line=dict(color='red', width=2),
        yaxis='y1',
        fill='tozeroy',
        fillcolor='rgba(239, 68, 68, 0.1)',
        #secondary_y=False
    ))

# 2. Precipitation - Bar trace (Second Y axis, right)
    fig.add_trace(go.Bar(
        x=hourly['date'],
        y=hourly['precipitation'],
        name="Precipitation (mm)",
        marker_color='dodgerblue',
        yaxis='y2',
        opacity=0.6,
        #secondary_y=False
    ))

# 3. Wind Speed - Broken line or scatter (Third Y axis, right/secondary)
    fig.add_trace(go.Scatter(
        x=hourly['date'],
        y=hourly['windspeed_10m'],
        name="Wind Speed (km/h)",
        mode='lines+markers',
        line=dict(color='green', width=2, dash='dash'),
        yaxis='y3',
        #secondary_y=True
    ))

# Axes setup
    fig.update_layout(
        xaxis=dict(
            title="Time",
            tickformat='%H:%M',
            dtick=3600000 * 3  # every 3 hours (adjust if needed)
        ),
        yaxis=dict(
            title="Temperature (°C)",
            color='red',
            range=[hourly['temperature_2m'].min() - 1, hourly['temperature_2m'].max() + 1]
        ),
        yaxis2=dict(
            title="Precipitation (mm) | Wind Speed (km/h)",
            overlaying="y",
            side="right",
            color='dodgerblue',
            range=[0, max(hourly['precipitation'].max(), hourly['windspeed_10m'].max()) + 1]
        ),
        yaxis3=dict(
            title="",
            anchor="x",
            overlaying="y",
            side="right",
            position=0.98,  # offset so it's not on top of y2
            color="green",
            range=[0, max(hourly['precipitation'].max(), hourly['windspeed_10m'].max()) + 1]
        ),
        legend=dict(orientation='h', yanchor='top', y=1, xanchor='left', x=0,  bgcolor='rgba(0,0,0,0)'),
        margin=dict(t=30, b=30),
        template="plotly_dark"
    )

    fig.update_xaxes(range=[hourly['date'].min() - timedelta(minutes=30), hourly['date'].max() + timedelta(minutes=30)])


# Render to Streamlit
    st.plotly_chart(fig, use_container_width=True)


    # -------------------------------
    # 10. WEATHER CORRELATION HEATMAP
    # -------------------------------

    # Define the weather variables to use in the analysis
    numeric_cols = [
        'temperature_2m',
        'relativehumidity_2m',
        'precipitation',
        'windspeed_10m',
        'apparent_temperature'
    ]
    corr_matrix = df[numeric_cols].corr()

    st.markdown("---")
    st.markdown(f"""<h2 style  = font-size: 40px;>Weather Variables & Impact Analysis</h2>""", unsafe_allow_html=True)
    

    col1, col2 = st.columns([1, 2])  # main layout

    with col1:
        selected_var1 = st.selectbox("Variable 1", numeric_cols, key="var1")
        selected_var2 = st.selectbox("Variable 2", numeric_cols, key="var2", index=1)
        if selected_var1 == selected_var2:
            st.error("Please select two different variables to compare.")
            st.stop()
        corr_value = corr_matrix.loc[selected_var1, selected_var2]
        abs_corr = abs(corr_value)

        if abs_corr < 0.2:
            label = "No correlation"
        elif abs_corr < 0.4:
            label = "Weak correlation"
        elif abs_corr < 0.7:
            label = "Moderate correlation"
        else:
            label = "Strong correlation" if corr_value > 0 else "Strong negative correlation"
        st.subheader(label)
        st.write(f"Correlation coefficient: {corr_value:.2f}")

        layman_descriptions = {
            ("temperature_2m", "precipitation"): "Rainfall often cools the air.",
            ("temperature_2m", "windspeed_10m"): "Stronger winds usually make it feel cooler.",
            ("precipitation", "windspeed_10m"): "Wind and rain together are typical of storms.",
            ("relativehumidity_2m", "temperature_2m"): "Hot and humid days feel especially muggy.",
            ("relativehumidity_2m", "precipitation"): "Humid air makes rain more likely.",
            ("apparent_temperature", "windspeed_10m"): "The wind can change the temperature you feel.",
        }
        key = tuple(sorted([selected_var1, selected_var2]))
        layman = layman_descriptions.get(key, "No clear pattern between these variables.")
        st.write(layman)

    with col2:
        numpoints = min(len(df), 1000)
        x = df[selected_var1][:numpoints]
        y = df[selected_var2][:numpoints]

        scatter = go.Figure()
        scatter.add_trace(go.Scattergl(
            x=x,
            y=y,
            mode='markers',
            marker=dict(size=7, opacity=0.7),
            name=f"{selected_var1} vs {selected_var2}"
        ))

        scatter.update_layout(
            xaxis_title=selected_var1.replace("_", " ").title(),
            yaxis_title=selected_var2.replace("_", " ").title(),
            title=f"{selected_var1} vs {selected_var2}",
            margin=dict(t=40, r=10, b=40, l=50),
            height=400,
            width=650
        )

        st.plotly_chart(scatter, use_container_width=True)
    # -------------------------------
    # 11. PDF REPORT DOWNLOAD
    # -------------------------------
    _ = """
    data = get_weather_and_suspension(datetime.combine(selected_date, datetime.min.time()), city, df, load_model_artifacts())
    if not data:
        st.warning("No data available for the selected date and city.")
        return
    
    weather_data = data['weather_data']

    # Now define your variables for PDF from weather_data
    temperature = weather_data.get("temperature_2m")
    humidity = weather_data.get("relativehumidity_2m")
    precip = weather_data.get("precipitation")
    feels_like = weather_data.get("apparent_temperature")
    wind_speed = weather_data.get("windspeed_10m")
    # Also from your Weather Variables & Impact Analysis logic, assign:
    selected_variable_pair = f"{selected_var1} vs {selected_var2}"
    correlation_label = label  # From your correlation label logic
    layman_text = layman        # From your layman descriptions lookup

    if st.button("Download PDF report"):
        temp_imgs = []
        try:
            # Export chart images to temporary files
            for fig_obj in [fig_gauge, fig_map, fig, scatter]:
                temp_img = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                pio.write_image(fig_obj, temp_img.name, format='png', width=500, height=270, engine='kaleido')
                temp_imgs.append(temp_img.name)
                temp_img.close()

            # Compose PDF with text AND images
            pdf = FPDF(orientation="P", unit="mm", format="A4")
            pdf.add_page()
            pdf.set_font("Arial", "B", size=18)
            pdf.cell(0, 15, "Herald Weather Analytics", ln=True, align="C")

            # Daily summary section
            pdf.set_font("Arial", "", 13)
            pdf.cell(0, 8, "Daily Summary", ln=True)
            pdf.multi_cell(0,8,
                f"Temperature: {temperature:.1f}°C  Humidity: {humidity:.0f}%  Precipitation: {precip:.1f}mm")
            pdf.multi_cell(0, 8, f"Feels Like: {feels_like:.1f}°C  Wind Speed: {wind_speed:.1f}km/h"   
            )
            pdf.ln(10)

            # Suspension gauge & map side by side (keeping same y for both)
            y_position = pdf.get_y()
            pdf.image(temp_imgs[0], x=10, y=y_position, w=90)
            pdf.image(temp_imgs[1], x=110, y=y_position, w=90)
            pdf.ln(65)

            # Hourly weather trends
            pdf.set_font("Arial", "B", 13)
            pdf.cell(0, 10, "Hourly Weather Trends", ln=True)
            pdf.image(temp_imgs[2], x=10, y=pdf.get_y(), w=pdf.w - 20)
            pdf.add_page()

            # Impact analysis
            pdf.set_font("Arial", "B", 13)
            pdf.cell(0, 10, "Weather Variables & Impact Analysis", ln=True)
            pdf.set_font("Arial", "", 11)
            pdf.multi_cell(
                0,
                8,
                f"Selected Variable Pair: {selected_variable_pair}\nCorrelation: {correlation_label}\nLayman: {layman_text}",
            )
            pdf.image(temp_imgs[3], x=10, y=pdf.get_y(), w=pdf.w - 20)
            pdf.ln(50)

            
            pdf.set_y(-40)
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 6, "HERALD v2.0", ln=True, align="C")
            pdf.set_font("Arial", "", 11)
            pdf.cell(0, 6, "Hydrometeorological Early Risk Assessment and Live Decision-support", ln=True, align="C")
            pdf.cell(0, 6, "Powered by LightGBM ML + PAGASA Criteria | Forecast weather data from Open-Meteo API", ln=True, align="C")

        
            temp_pdf = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
            pdf.output(temp_pdf.name)

            with open(temp_pdf.name, "rb") as f:
                pdf_bytes = f.read()


            st.download_button("Download PDF", pdf_bytes, file_name="Weather_Analytics_Report.pdf", mime="application/pdf")


        finally:
                # Clean up temporary image files
             for path in temp_imgs:
                if os.path.exists(path):
                    os.remove(path)
    """

# ============================================================================
# PAGE: HISTORICAL ANALYTICS
# ============================================================================

def page_historical_analytics():
    st.markdown(
    "<h1 style='font-size: 40px;'>⌛Historical Analytics</h1>",
    unsafe_allow_html=True
)
    
    df = load_historical_data()
    min_date, max_date = get_date_bounds()
    
    with st.sidebar:
        st.markdown("### Selection Settings")
        col1, col2 = st.columns(2)
        
        # Initialize session state for dates if not exists
        if 'hist_start_date' not in st.session_state:
            st.session_state.hist_start_date = datetime(2025, 7, 20).date()
        if 'hist_end_date' not in st.session_state:
            st.session_state.hist_end_date = datetime(2025, 9, 12).date()
        
        with col1:
            # Start date picker: end date and beyond are disabled
            start_date = st.date_input(
                "Start Date", 
                value=st.session_state.hist_start_date,
                min_value=min_date, 
                max_value=st.session_state.hist_end_date,  # Can't go past end date
                key='hist_start'
            )
            st.session_state.hist_start_date = start_date
        
        with col2:
            # End date picker: start date and before are disabled
            end_date = st.date_input(
                "End Date", 
                value=st.session_state.hist_end_date,
                min_value=st.session_state.hist_start_date,  # Can't go before start date
                max_value=max_date,
                key='hist_end'
            )
            st.session_state.hist_end_date = end_date
        
        selected_cities = st.multiselect("Select Cities", sorted(CITY_COORDINATES.keys()), default=['Manila', 'Quezon City'], key='hist_cities')
    
    mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
    if selected_cities:
        mask = mask & (df['city'].isin(selected_cities))
    filtered_df = df[mask]
    
    if len(filtered_df) == 0:
        st.warning("No data available for selected filters.")
        return
    
    st.markdown("---")
    st.markdown("<h3 style='font-size: 40px;'>Summary Statistics</h3>", unsafe_allow_html=True)

    st.markdown("""
    <style>
    .stMetric-label {
        font-size: 20px !important;
    }
    .stMetric-value {
        font-size: 28px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(filtered_df):,}")
    with col2:
        st.metric("Suspension Rate", f"{(filtered_df['suspension'] > 0).mean() * 100:.1f}%")
    with col3:
        st.metric("Avg Precipitation", f"{filtered_df['precipitation'].mean():.1f}mm")
    with col4:
        st.metric("Avg Temperature", f"{filtered_df['temperature_2m'].mean():.1f}°C")

    st.markdown("---")
    st.markdown(f"""<h2 style  = font-size: 40px;>Suspension Level Distribution </h2>""", unsafe_allow_html=True)
    
    
    suspension_counts = filtered_df.groupby('suspension')['date'].apply(lambda x: x.dt.date.nunique()).reset_index()
    suspension_counts.columns = ['suspension', 'days']
    suspension_counts = suspension_counts[suspension_counts['suspension'] < 5].sort_values('suspension')
    
    fig = go.Figure(go.Bar(
        x=[f"Level {int(i)}" for i in suspension_counts['suspension']], 
        y=suspension_counts['days'], 
        marker_color=[SUSPENSION_LEVELS[int(i)]['color'] for i in suspension_counts['suspension']], 
        text=suspension_counts['days'], 
        texttemplate='%{text}', 
        textposition='auto'
    ))
    fig.update_layout(xaxis_title="Suspension Level", yaxis_title="Number of Days", height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown(f"""<h2 style  = font-size: 40px;>City-wise Suspension Analysis </h2>""", unsafe_allow_html=True)
    
    
    city_suspension = filtered_df.groupby('city')['suspension'].agg(['mean', 'count']).reset_index().sort_values('mean', ascending=False)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=city_suspension['city'], y=city_suspension['mean'], name='Avg Suspension Level', marker_color='#3b82f6', yaxis='y', offsetgroup=1))
    fig.add_trace(go.Scatter(x=city_suspension['city'], y=city_suspension['count'], name='Record Count', marker_color='#10b981', yaxis='y2', mode='lines+markers'))
    fig.update_layout(xaxis_title="City", yaxis_title="Avg Suspension Level", yaxis2=dict(title="Record Count", overlaying='y', side='right'), height=500, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown(f"""<h2 style  = font-size: 40px;>Weather-Suspension Correlation </h2>""", unsafe_allow_html=True)
    
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(filtered_df, x='suspension', y='precipitation', title='Precipitation by Suspension Level', color='suspension', color_discrete_map={i: SUSPENSION_LEVELS[i]['color'] for i in range(5)})
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(filtered_df, x='suspension', y='windspeed_10m', title='Wind Speed by Suspension Level', color='suspension', color_discrete_map={i: SUSPENSION_LEVELS[i]['color'] for i in range(5)})
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        fig = px.box(filtered_df, x='suspension', y='temperature_2m', title='Temperature by Suspension Level', color='suspension', color_discrete_map={i: SUSPENSION_LEVELS[i]['color'] for i in range(5)})
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.box(filtered_df, x='suspension', y='relativehumidity_2m', title='Humidity by Suspension Level', color='suspension', color_discrete_map={i: SUSPENSION_LEVELS[i]['color'] for i in range(5)})
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown(f"""<h2 style  = font-size: 40px;>Temporal Trends </h2>""", unsafe_allow_html=True)
    
    
    daily_suspension = filtered_df.groupby(filtered_df['date'].dt.date)['suspension'].mean().reset_index()
    daily_suspension.columns = ['date', 'avg_suspension']
    
    fig = px.line(daily_suspension, x='date', y='avg_suspension', title='Average Suspension Level Over Time', markers=True, line_shape='linear')
    fig.update_layout(xaxis_title='Date', yaxis_title='Avg Suspension Level', height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown(f"""<h2 style  = font-size: 40px;>Hour vs Day of Week Heatmap </h2>""", unsafe_allow_html=True)
    
    
    heatmap_variable = st.selectbox(
        "Select Weather Variable",
        ["Temperature (°C)", "Precipitation (mm)", "Wind Speed (km/h)", "Humidity (%)"],
        key='heatmap_var'
    )
    
    filtered_df_copy = filtered_df.copy()
    filtered_df_copy['hour'] = filtered_df_copy['date'].dt.hour
    filtered_df_copy['day_of_week'] = filtered_df_copy['date'].dt.day_name()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    filtered_df_copy['day_of_week'] = pd.Categorical(filtered_df_copy['day_of_week'], categories=day_order, ordered=True)
    
    if heatmap_variable == "Temperature (°C)":
        heatmap_data = filtered_df_copy.pivot_table(values='temperature_2m', index='day_of_week', columns='hour', aggfunc='mean')
        colorscale = 'RdYlBu_r'
        title = 'Average Temperature by Hour and Day of Week'
    elif heatmap_variable == "Precipitation (mm)":
        heatmap_data = filtered_df_copy.pivot_table(values='precipitation', index='day_of_week', columns='hour', aggfunc='mean')
        colorscale = 'Blues'
        title = 'Average Precipitation by Hour and Day of Week'
    elif heatmap_variable == "Wind Speed (km/h)":
        heatmap_data = filtered_df_copy.pivot_table(values='windspeed_10m', index='day_of_week', columns='hour', aggfunc='mean')
        colorscale = 'Greens'
        title = 'Average Wind Speed by Hour and Day of Week'
    else:
        heatmap_data = filtered_df_copy.pivot_table(values='relativehumidity_2m', index='day_of_week', columns='hour', aggfunc='mean')
        colorscale = 'Purples'
        title = 'Average Humidity by Hour and Day of Week'
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data.values,
        x=heatmap_data.columns,
        y=heatmap_data.index,
        colorscale=colorscale,
        text=heatmap_data.values,
        texttemplate='%{text:.1f}',
        textfont={"size": 9},
        colorbar=dict(title=heatmap_variable),
        hoverongaps=False
    ))
    
    fig_heatmap.update_layout(
        title=title,
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        height=500,
        width=1000
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)

# ============================================================================
# PAGE: WHAT-IF SCENARIOS
# ============================================================================

def page_what_if():
    st.markdown(
    "<h1 style='font-size: 40px;'>🤔💭What-If Scenario Predictor</h1>",
    unsafe_allow_html=True
)
    st.markdown("### Adjust weather parameters to see predicted suspension levels based on ML model and PAGASA criteria")
    
    artifacts = load_model_artifacts()
    
    with st.sidebar:
        st.markdown("### Weather Parameters")
        city = st.selectbox("Select City", sorted(CITY_COORDINATES.keys()), key='what_if_city')
        temperature = st.slider("Temperature (°C)", 20.0, 55.0, 28.0, 0.5)
        humidity = st.slider("Relative Humidity (%)", 30, 100, 75, 1)
        precipitation = st.slider("Precipitation (mm)", 0.0, 100.0, 10.0, 1.0)
        wind_speed = st.slider("Wind Speed (km/h)", 0, 300, 20, 1)
        
        st.markdown("### Temporal Context")
        hour = st.slider("Hour of Day", 0, 23, 12, key='what_if_hour')
        day_of_week = st.selectbox("Day of Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'], key='what_if_dow')
        month = st.selectbox("Month", list(range(1, 13)), index=6, key='what_if_month')
    
    weather_data = {
        'temperature_2m': temperature,
        'relativehumidity_2m': humidity,
        'precipitation': precipitation,
        'apparent_temperature': temperature - 2,
        'windspeed_10m': wind_speed
    }
    
    hourly_df = pd.DataFrame({
        'date': [datetime.now().replace(hour=hour, month=month)],
        'temperature_2m': [temperature],
        'relativehumidity_2m': [humidity],
        'precipitation': [precipitation],
        'apparent_temperature': [temperature - 2],
        'windspeed_10m': [wind_speed]
    })
    
    with st.spinner("Analyzing weather conditions..."):
        ml_prediction = predict_suspension(weather_data, city, hourly_df, artifacts)
        pagasa_level, pagasa_reasons = check_pagasa_criteria(weather_data)
        
        st.markdown("---")
        st.markdown(f"""<h2 style  = font-size: 40px;>Prediction Results </h2>""", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            final_level = max(ml_prediction['predicted_level'], pagasa_level)
            fig = create_suspension_gauge(final_level, True, ml_prediction['confidence'])
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            info = SUSPENSION_LEVELS[final_level]
            st.markdown(f"<div style='padding: 30px; font-size:20px; background-color: {info['color']}20; border-left: 4px solid {info['color']}; border-radius: 5px;'><h2 style='margin:0; color: {info['color']};'>{info['emoji']} {info['name']}</h2><p style='margin: 10px 0 0 0;'><strong>Risk Level:</strong> {info['risk']}</p></div>", unsafe_allow_html=True)
            
            st.markdown(f"""<h3 style = font-size:20 px;> Suspension Justification</h3>""", unsafe_allow_html=True)
            st.markdown(f"""<p style = font-size:20px;>LightGBM ML Model: Level {ml_prediction['predicted_level']} (Confidence: {ml_prediction['confidence']*100:.1f}%)</p>""", unsafe_allow_html=True)
            
            if pagasa_level > 0:
                st.markdown(f"**PAGASA Criteria**: Level {pagasa_level}")
                for reason in pagasa_reasons:
                    st.markdown(f"- {reason}")
            else:
                st.markdown("**PAGASA Criteria**: No suspension recommended")
            
            if final_level > 0:
                st.success(f"Suspension Recommended: Level {final_level}")
                if pagasa_level > ml_prediction['predicted_level']:
                    st.info("Recommendation based on PAGASA's official criteria")
                else:
                    st.info("Recommendation based on LightGBM model classification")
            else:
                st.success("No Suspension Recommended")
        
        st.markdown("---")
        st.markdown(f"""<h2 style  = font-size: 40px;>Probability Distribution (ML Model)</h2>""", unsafe_allow_html=True)
        
        probs = ml_prediction['probabilities']
        if len(probs) < 5:
            probs = list(probs) + [0.0] * (5 - len(probs))
        else:
            probs = list(probs)[:5]
        
        prob_df = pd.DataFrame({'Level': [f"Level {i}" for i in range(5)], 'Probability': [p * 100 for p in probs]})
        fig = go.Figure(go.Bar(x=prob_df['Level'], y=prob_df['Probability'], marker_color=[SUSPENSION_LEVELS[i]['color'] for i in range(len(prob_df))], text=[f"{p:.1f}%" for p in prob_df['Probability']], textposition='auto'))
        fig.update_layout(xaxis_title="Suspension Level", yaxis_title="Probability (%)", height=300, margin=dict(l=20, r=20, t=20, b=50))
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE: ABOUT
# ============================================================================

def page_about():
    st.markdown(
    "<h1 style='font-size: 40px;'>⚙️About this System</h1>",
    unsafe_allow_html=True
)
    
    artifacts = load_model_artifacts()
    
    st.markdown("<h2 style='font-size: 2.5rem;'>Metro Manila Suspension Prediction System</h2>", unsafe_allow_html=True)

    st.markdown(
        """
        <p style='font-size: 1.3rem;'>
        <em>HERALD v2.0</em> (Hydrometeorological Early Risk Assessment and Live Decision-support) is an intelligent 
        system that predicts class and work suspension levels in Metro Manila based on weather conditions and official 
        meteorological criteria.
        </p>
        """, unsafe_allow_html=True)

    st.markdown("<h3 style='font-size: 2rem;'>How It Works</h3>", unsafe_allow_html=True)

    steps = [
        "*Historical Mode*: When you select a date within our dataset range, the system displays actual recorded weather data and historical suspension decisions.",
        "*Forecast Mode*: For future dates, the system fetches weather forecasts from the Open-Meteo API and uses our trained ML model to predict suspension levels.",
        "*What-If Scenarios*: Adjust weather parameters manually to explore how different conditions affect suspension predictions using both ML and PAGASA criteria.",
        "*Weather Analytics*: Detailed analysis of weather patterns for specific dates and cities.",
        "*Historical Analytics*: Comprehensive analytics of suspension trends across date ranges and multiple cities."
    ]

    for step in steps:
        st.markdown(f"<p style='font-size: 1.1rem;'>{step}</p>", unsafe_allow_html=True)

    st.markdown("<h3 style='font-size: 2rem;'>Machine Learning Model</h3>", unsafe_allow_html=True)

    trainings = [
        "*Stratified Train-Test Split*: Ensures all suspension levels are represented in both training and testing",
        "*Aggressive Oversampling*: Minority classes are oversampled to 80% of majority class size with noise injection",
        "*Class Weighting*: Higher penalties for misclassifying rare suspension levels",
        "*Hyperparameter Optimization*: Tuned for maximum depth and complexity to capture rare patterns"
    ]

    for train in trainings:
        st.markdown(f"<p style='font-size: 1.1rem;'>{train}</p>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", "LightGBM Classifier")
        st.metric("Total Records", f"{artifacts['metrics']['dataset_info']['total_records']:,}")
    
    with col2:
        # Show both weighted and balanced accuracy
        weighted_acc = artifacts['metrics']['models']['lightgbm_tuned']['accuracy']
        balanced_acc = artifacts['metrics']['models']['lightgbm_tuned'].get('balanced_accuracy', weighted_acc)
        st.metric("Weighted Accuracy", f"{weighted_acc*100:.2f}%")
        st.metric("Balanced Accuracy", f"{balanced_acc*100:.2f}%", 
                 help="Accounts for class imbalance - more important metric for this dataset")
    
    with col3:
        st.metric("Cities Covered", len(artifacts['metrics']['dataset_info']['cities']))
        feature_count = len(artifacts['metrics'].get('feature_names', []))
        if feature_count == 0:
            feature_count = len(artifacts['feature_importance'])
        st.metric("Features Used", feature_count)
    
    st.markdown("---")
    st.markdown("<h1 style='font-size:40px; font-weight:bold;'>Model Performance by Class</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:25px; color:gray;'>Handling extreme class imbalance with specialized techniques</p>", unsafe_allow_html=True)

    
    metrics = artifacts['metrics']['models']['lightgbm_tuned']
    
    # Create per-class performance table
    classes = list(range(len(metrics['precision_per_class'])))
    performance_data = {
        'Suspension Level': [f"Level {i}" for i in classes],
        'Precision': [f"{p*100:.2f}%" for p in metrics['precision_per_class']],
        'Recall': [f"{r*100:.2f}%" for r in metrics['recall_per_class']],
        'F1 Score': [f"{f*100:.2f}%" for f in metrics['f1_per_class']],
        'Support': metrics.get('support_per_class', [0]*len(classes))
    }
    
    perf_df = pd.DataFrame(performance_data)
    
    # Color code the metrics
    def color_metric(val):
        if isinstance(val, str) and '%' in val:
            num = float(val.replace('%', ''))
            if num >= 80:
                return 'background-color: #d1fae5'  # Green
            elif num >= 50:
                return 'background-color: #fef3c7'  # Yellow
            elif num >= 20:
                return 'background-color: #fed7aa'  # Orange
            else:
                return 'background-color: #fecaca'  # Red
        return ''
    
    styled_df = perf_df.style.applymap(color_metric, subset=['Precision', 'Recall', 'F1 Score'])
    st.dataframe(styled_df, use_container_width=True)
    
    # Add explanation
    st.info("""
    *Understanding the Metrics:*
    - 🟢 Green (≥80%): Excellent performance
    - 🟡 Yellow (50-80%): Good performance  
    - 🟠 Orange (20-50%): Fair performance - model learning patterns
    - 🔴 Red (<20%): Poor performance - extremely rare events
    
    *Support* shows the number of actual instances in the test set. Lower support means fewer examples to learn from.
    """)
    
    st.markdown("---")
    st.markdown("<h2 style='font-size:40px; font-weight:bold;'>Aggregate Performance Metrics</h2>", unsafe_allow_html=True)

    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h2 style='font-size:30px; font-weight:bold;'>Weighted Metrics</h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:25px; color:gray;'>Accounts for class frequency in dataset</p>", unsafe_allow_html=True)
        st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
        st.metric("Precision", f"{metrics['precision']*100:.2f}%")
        st.metric("Recall", f"{metrics['recall']*100:.2f}%")
        st.metric("F1 Score", f"{metrics['f1']*100:.2f}%")
    
    with col2:
        st.markdown("<h2 style='font-size:30px; font-weight:bold;'>Macro Metrics</h2>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:25px; color:gray;'>Treats all classes equally (better for imbalanced data)</p>", unsafe_allow_html=True)

        macro_precision = metrics.get('macro_precision', 0)
        macro_recall = metrics.get('macro_recall', 0)
        macro_f1 = metrics.get('macro_f1', 0)
        balanced_acc = metrics.get('balanced_accuracy', 0)
        
        st.metric("Balanced Accuracy", f"{balanced_acc*100:.2f}%")
        st.metric("Macro Precision", f"{macro_precision*100:.2f}%")
        st.metric("Macro Recall", f"{macro_recall*100:.2f}%")
        st.metric("Macro F1 Score", f"{macro_f1*100:.2f}%")
    
    st.markdown("---")
    st.markdown("<h1 style='font-size:40px; font-weight:bold;'>Feature Importance</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:25px; color:gray;'>Top features that influence suspension predictions</p>", unsafe_allow_html=True)

    
    feature_imp = artifacts['feature_importance']
    top_features = dict(list(feature_imp.items())[:15])
    
    fig = go.Figure(go.Bar(
        x=list(top_features.values()), 
        y=list(top_features.keys()), 
        orientation='h', 
        marker_color='#009cdf', 
        text=[f"{v:.0f}" for v in top_features.values()], 
        textposition='outside',
        outsidetextfont=dict(size=40)
    ))
    fig.update_layout(
        xaxis_title="Importance Score",
        yaxis_title="Feature",
        height=500,
        yaxis=dict(
            categoryorder='total ascending',
            title_font=dict(size=17),
            tickfont=dict(size=17)
        ),
        xaxis=dict(
            title_font=dict(size=17),
            tickfont=dict(size=17))
)


    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    st.markdown("<h1 style='font-size:40px; font-weight:bold;'>Training Configuration</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:25px; color:gray;'>Predicted vs Actual Suspension Levels (Test Set)</p>", unsafe_allow_html=True)
    
    cm = np.array(metrics['confusion_matrix'])
    
    # Normalize confusion matrix for better visualization
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  
    
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm_normalized, 
        x=[f"Level {i}" for i in range(len(cm))], 
        y=[f"Level {i}" for i in range(len(cm))], 
        colorscale='Blues', 
        text=cm,
        customdata=cm_normalized,
        hovertemplate='Actual: %{y}<br>Predicted: %{x}<br>Count: %{text}<br>Rate: %{customdata:.1%}<extra></extra>',
        texttemplate='%{text}', 
        textfont={"size": 12}, 
        colorbar=dict(title="Accuracy<br>Rate"),
        hoverongaps=False
    ))
    fig_cm.update_layout(
        xaxis_title="Predicted Level", 
        yaxis_title="Actual Level", 
        height=500, 
        width=600
    )
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.plotly_chart(fig_cm, use_container_width=True)
    
    with col2:
        st.markdown("### Cross-Validation")
        st.caption("5-fold CV on balanced training data")
        st.metric("CV Mean", f"{metrics['cv_mean']*100:.2f}%")
        st.metric("CV Std", f"±{metrics['cv_std']*100:.2f}%")
        
        st.markdown("---")
        st.markdown("### Class Imbalance Handling")
        st.markdown("""
        - ✅ Stratified splitting
        - ✅ 80% oversampling target
        - ✅ Noise injection (2%)
        - ✅ Inverse frequency weighting
        - ✅ Deep trees (depth=10)
        - ✅ 800 estimators
        """)
    
    st.markdown("---")
    st.markdown("<h1 style='font-size:40px; font-weight:bold;'>Training Configuration</h1>", unsafe_allow_html=True)
    
    best_params = metrics.get('best_params', {})
    if best_params:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("<p style='font-size:20px; font-weight:bold;'>Learning Parameters</p>", unsafe_allow_html=True)
            st.code(f"""
learning_rate: {best_params.get('learning_rate', 'N/A')}
n_estimators: {best_params.get('n_estimators', 'N/A')}
max_depth: {best_params.get('max_depth', 'N/A')}
num_leaves: {best_params.get('num_leaves', 'N/A')}
            """)
        
        with col2:
            st.markdown("<p style='font-size:20px; font-weight:bold;'>Regularization</p>", unsafe_allow_html=True)
            st.code(f"""
reg_alpha: {best_params.get('reg_alpha', 'N/A')}
reg_lambda: {best_params.get('reg_lambda', 'N/A')}
min_child_samples: {best_params.get('min_child_samples', 'N/A')}
            """)
        
        with col3:
            st.markdown("<p style='font-size:20px; font-weight:bold;'>Sampling Parameters</p>", unsafe_allow_html=True)

            st.code(f"""
subsample: {best_params.get('subsample', 'N/A')}
colsample_bytree: {best_params.get('colsample_bytree', 'N/A')}
boost_from_average: {best_params.get('boost_from_average', 'N/A')}
            """)
    
    st.markdown(f"""<h1 style = font-size: 40px;>Dataset Information</h1>""", unsafe_allow_html=True)
    st.markdown(f"""<p style ='font-size: 25px;'>Date Range: 2020-01-01 to 2025-10-10</p>""", unsafe_allow_html=True)
    st.markdown(f"""<p style ='font-size: 25px;'>Cities Covered</p>""", unsafe_allow_html=True)
    st.markdown(f"""<p style ='font-size: 20px;'>
                <a href='https://www.facebook.com/caloocan.pio/' target='_blank' title='Caloocan Facebook Page'>Caloocan</a>, 
                <a href='https://www.facebook.com/cityoflaspinasofficial/' target='_blank' title='Las Piñas Facebook Page'>Las Piñas</a>, 
                <a href='https://www.facebook.com/MyMakatiVerified/' target='_blank' title='Makati Facebook Page'>Makati</a>, 
                <a href='https://www.facebook.com/MalabonCityGov/' target='_blank' title='Malabon Facebook Page'>Malabon</a>, 
                <a href='https://www.facebook.com/MandaluyongPIO/' target='_blank' title='Mandaluyong Facebook Page'>Mandaluyong</a>,
                <a href='https://www.facebook.com/ManilaPIO/' target='_blank' title='Manila Facebook Page'>Manila</a>, 
                <a href='https://www.facebook.com/MarikinaPIO/' target='_blank' title='Marikina Facebook Page'>Marikina</a>, 
                <a href='https://www.facebook.com/officialMuntinlupacity/' target='_blank' title='Muntinlupa Facebook Page'>Muntinlupa</a>, 
                <a href='https://www.facebook.com/navotenoako/' target='_blank' title='Navotas Facebook Page'>Navotas</a>, 
                <a href='https://www.facebook.com/pioparanaque' target='_blank' title='Parañaque Facebook Page'>Parañaque</a>,
                <a href='https://www.facebook.com/lgupasaypio' target='_blank' title='Pasay Facebook Page'>Pasay</a>, 
                <a href='https://www.facebook.com/PasigPIO' target='_blank' title='Pasig Facebook Page'>Pasig</a>, 
                <a href='https://www.facebook.com/municipalityofpateros' target='_blank' title='Pateros Facebook Page'>Pateros</a>, 
                <a href='https://www.facebook.com/QCGov/' target='_blank' title='Quezon City Facebook Page'>Quezon City</a>, 
                <a href='https://www.facebook.com/MayorFrancisZamora' target='_blank' title='San Juan Facebook Page'>San Juan</a>,
                <a href='https://www.facebook.com/taguigcity/?locale=tl_PH' target='_blank' title='Taguig Facebook Page'>Taguig</a>, 
                <a href='https://www.facebook.com/ValenzuelaCityGov' target='_blank' title='Valenzuela Facebook Page'>Valenzuela</a> </p>""", unsafe_allow_html=True)
    st.markdown(f"""<p style='font-size: 25px;'> Split Strategy (80% train, 20% test)</p>""", unsafe_allow_html=True)
 
    st.markdown(f"""<h1 style = 'font-size: 40px;'>Class Distribution</h1>""", unsafe_allow_html=True)
    st.markdown(f"""<p style =  'font-size: 25px;'>The dataset exhibits extreme class imbalance, with Level 0 (No Suspension) comprising ~98% of records. Our model uses specialized techniques to ensure all suspension levels are learned effectively.</p>""", unsafe_allow_html=True)
    st.markdown(f"""<h1 style = 'font-size: 40px;'>Suspension Levels Explained</h1>""", unsafe_allow_html=True)

    # For your suspension levels below, you can also do something similar:
    for level, info in SUSPENSION_LEVELS.items():
        st.markdown(f"<p style='font-size:25px;'>*{info['emoji']} Level {level}: {info['name']}* | Risk: {info['risk']}</p>", unsafe_allow_html=True)

    
    st.markdown("<h1 style='font-size:40px;'>PAGASA Suspension Criteria</h1>", unsafe_allow_html=True)

    st.markdown("<p style='font-size:20px;'>The system also checks against official PAGASA (Philippine Atmospheric, Geophysical and Astronomical Services Administration) criteria for suspension decisions:</p>", unsafe_allow_html=True)


    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div style='font-size:1.2rem;'>
                <b>Wind-based (Typhoon Warning Signal)</b><br>
                TCWS 1: 30-60 km/h → Preschool Suspension<br>
                TCWS 2: 60-100 km/h → Preschool + Elementary Suspension<br>
                TCWS 3: 100-150 km/h → All School Levels Suspension<br>
                TCWS 4: >150 km/h → All Levels + Work Suspension
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style='font-size:1.2rem;'>
                <b>Rainfall-based</b><br>
                Light: ≥7.5 mm/hour → Preschool Suspension<br>
                Moderate: ≥15 mm/hour → All School Levels Suspension<br>
                Heavy: ≥30 mm/hour → All Levels + Work Suspension
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div style='font-size:1.2rem;'>
                <b>Heat Index-based</b><br>
                Heat Index 41°C → All School Levels Suspended<br>
                Heat Index 51°C → All Levels + Work Suspended
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<h1 style='font-size:40px;'>Development Team</h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:25px; color:gray;'>Meet the team behind HERALD v2.0</p>", unsafe_allow_html=True)

    
    
    team_members = [
        {
            "name": "Gabrielle B. Cabanilla",
            "school": "Polytechnic University of the Philippines",
            "description": "A Railway Engineering graduate who’s always been fascinated by how movement connects people and places. He thrives on solving real-world problems through innovation, precision, and teamwork — always aiming to build systems that move the world forward. ",
            "contact": "cabanillla.gb@gmail.com",
            "image": "assets/gab.jpg"
        },
        {
           "name": "Jeremy Charles B. Mora ",
            "school": "Technological Institute of the Philippines",
            "description": "A Computer Science major who loves turning data into meaningful insights. Curious and analytical, he enjoys transforming raw information into stories that drive smarter decisions. From data cleaning to visualization and predictive modeling, he’s always exploring new tools and technologies to uncover patterns, solve problems, and make complex information easier to understand.",
            "contact": "jeremycharlesmora@gmail.com",
            "image": "assets/morsi.jpg"
        },
        {
            "name": "John Trixie M. Ocampo",
            "school": "Pamantasan ng Lungsod ng Maynila",
            "description": "A BS Mathematics graduate who loves exploring patterns, logic, and the beauty behind numbers. Curious by nature and detail-oriented, he enjoys finding order in complexity and turning data into meaningful insights.",
            "contact": "jtocampo0118@gmail.com",
            "image": "assets/jtrixie.jpg"
            
        }
    ]
    
    # Create team members table
    for i, member in enumerate(team_members):
        with st.container():
            col1, col2 = st.columns([1, 3.5])

        with col1:
            image_path = member['image']
            if image_path.endswith('.jpg') or image_path.endswith('.png'):
                if os.path.exists(image_path):
                    st.image(image_path, width=200)
                else:
                    st.warning(f"Image not found: {image_path}")
            else:
                st.markdown(
                    f"<div style='text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin: 10px 0;'>"
                    f"<h1 style='font-size: 4rem; margin: 0;'>{member['image']}</h1>"
                    f"</div>", unsafe_allow_html=True)


            with col2:
                st.markdown(f"<h2 style='font-size: 30px;'>{member['name']}</h2>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 20px; font-weight: bold;'>🏫 {member['school']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 18px;'>{member['description']}</p>", unsafe_allow_html=True)
                st.markdown(f"<p style='font-size: 18p;'>📧 <strong>Contact:</strong> {member['contact']}</p>", unsafe_allow_html=True)

            if i < len(team_members) - 1:
                st.markdown("---")

    st.markdown("---")
    st.markdown(
        """
        <h3 style='text-align: center;'>Disclaimer</h3>
        <div style='text-align: center; font-size: 15px;'>
            This is a prediction tool and should not be used as the sole basis for official suspension decisions.<br>
            Always refer to official announcements from DepEd, DOST-PAGASA, and local government units.<br><br>
            <hr>
            <b>Version</b>: 2.0 | <b>Last Updated</b>: October 2025 | <b>Powered by</b>: LightGBM ML + PAGASA Criteria
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    try:
        if not os.path.exists('model.pkl'):
            st.error("Model not found! Please run 'python train_model.py' first.")
            st.info("Make sure you have run the training script to generate model artifacts.")
            st.code("python train_model.py", language="bash")
            return
        
        pages = {
            "Home": page_home,
            "Weather Analytics": page_weather_analytics,
            "Historical Analytics": page_historical_analytics,
            "What-If Scenarios": page_what_if,
            "About": page_about
        }

        with st.sidebar:
            # Initialize session state for selected page if not exists
            if 'selected_page' not in st.session_state:
                st.session_state.selected_page = "Home"
            
            # Page Navigation with expander
            with st.expander("📍 Page Navigation", expanded=True):
                # Create clickable menu items
                for page_name in pages.keys():
                    if st.button(
                        page_name, 
                        key=f"nav_{page_name}",
                        use_container_width=True,
                        type="primary" if st.session_state.selected_page == page_name else "secondary"
                    ):
                        st.session_state.selected_page = page_name
                        st.rerun()
            
            selected_page = st.session_state.selected_page

        
        pages[selected_page]()
        
        st.markdown("---")
        
        st.markdown("<div style='text-align: center; color: #666; padding: 20px;'><p><strong>HERALD v2.0</strong></p><p>Hydrometeorological Early Risk Assessment and Live Decision-support</p><p>Powered by LightGBM ML + PAGASA Criteria | Forecast weather data from Open-Meteo API</p></div>", unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)



if __name__ == "__main__":
    main()