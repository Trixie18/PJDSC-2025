"""
Metro Manila Class/Work Suspension Prediction System
A Streamlit application for predicting and analyzing suspension declarations
HERALD v2.0 - Enhanced with PAGASA criteria and mobile-responsive UI
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
    initial_sidebar_state="auto"  # Auto-collapse on mobile
)

# ============================================================================
# MOBILE-RESPONSIVE CSS
# ============================================================================

def inject_responsive_css():
    """Inject responsive CSS for mobile optimization"""
    st.markdown("""
    <style>
    /* Base responsive improvements */
    @media (max-width: 768px) {
        /* Reduce padding on mobile */
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            padding-top: 1rem !important;
        }
        
        /* Make headers responsive */
        h1 {
            font-size: 2rem !important;
        }
        
        h2 {
            font-size: 1.5rem !important;
        }
        
        h3 {
            font-size: 1.25rem !important;
        }
        
        /* Adjust metric font sizes */
        [data-testid="stMetricValue"] {
            font-size: 1.5rem !important;
        }
        
        [data-testid="stMetricLabel"] {
            font-size: 0.9rem !important;
        }
        
        /* Make buttons full width on mobile */
        .stButton > button {
            width: 100% !important;
        }
        
        /* Adjust sidebar width */
        [data-testid="stSidebar"] {
            min-width: 100% !important;
        }
        
        /* Stack columns on mobile */
        [data-testid="column"] {
            width: 100% !important;
            flex: 100% !important;
            max-width: 100% !important;
        }
        
        /* Improve plotly chart responsiveness */
        .js-plotly-plot {
            width: 100% !important;
        }
        
        /* Reduce image sizes */
        img {
            max-width: 100% !important;
            height: auto !important;
        }
    }
    
    /* Tablet responsiveness */
    @media (min-width: 769px) and (max-width: 1024px) {
        .main .block-container {
            padding-left: 2rem !important;
            padding-right: 2rem !important;
        }
        
        h1 {
            font-size: 2.5rem !important;
        }
    }
    
    /* Improve metric card styling */
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Better expander styling */
    .streamlit-expanderHeader {
        font-size: 1.1rem !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

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
# UTILITY FUNCTIONS FOR MOBILE DETECTION
# ============================================================================

def is_mobile():
    """Detect if user is on mobile device (simple heuristic based on screen width)"""
    # This is a simple check - in production you'd use JavaScript
    return st.session_state.get('is_mobile', False)

def get_responsive_columns(desktop_spec, mobile_spec=None):
    """
    Return column specification based on device type
    desktop_spec: list/tuple for desktop (e.g., [1, 2])
    mobile_spec: list/tuple for mobile (e.g., [1]) - defaults to single column
    """
    if mobile_spec is None:
        mobile_spec = [1] * len(desktop_spec)
    
    # For simplicity, always use mobile_spec for narrow layouts
    # In production, you'd check actual viewport width
    return desktop_spec

def create_responsive_header(text, mobile_size="1.5rem", desktop_size="3rem"):
    """Create responsive header"""
    st.markdown(
        f"<h1 style='font-size: clamp({mobile_size}, 5vw, {desktop_size});'>{text}</h1>",
        unsafe_allow_html=True
    )

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
# [REST OF THE UTILITY FUNCTIONS REMAIN THE SAME]
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
# MOBILE-OPTIMIZED VISUALIZATION FUNCTIONS
# ============================================================================

def create_suspension_gauge(level: int, is_prediction: bool = False, confidence: float = None):
    """Mobile-responsive gauge chart"""
    info = SUSPENSION_LEVELS[level]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=level,
        title={'text': f"{'Predicted' if is_prediction else 'Actual'}<br>{info['name']}", 'font': {'size': 14}},
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
    fig.update_layout(
        height=250, 
        margin=dict(l=10, r=10, t=60, b=10),
        font=dict(size=12)
    )
    return fig

def create_hourly_weather_chart(hourly_df: pd.DataFrame, mode: str):
    """Mobile-responsive hourly weather chart"""
    fig = make_subplots(
        rows=3, cols=1, 
        subplot_titles=('Temperature (°C)', 'Precipitation (mm)', 'Wind Speed (km/h)'),
        vertical_spacing=0.12
    )
    
    fig.add_trace(
        go.Scatter(
            x=hourly_df['date'], 
            y=hourly_df['temperature_2m'], 
            name='Temperature', 
            line=dict(color='#ef4444', width=1.5), 
            fill='tozeroy', 
            fillcolor='rgba(239, 68, 68, 0.1)'
        ), 
        row=1, col=1
    )
    fig.add_trace(
        go.Bar(
            x=hourly_df['date'], 
            y=hourly_df['precipitation'], 
            name='Precipitation', 
            marker_color='#3b82f6'
        ), 
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(
            x=hourly_df['date'], 
            y=hourly_df['windspeed_10m'], 
            name='Wind Speed', 
            line=dict(color='#10b981', width=1.5)
        ), 
        row=3, col=1
    )
    
    fig.update_xaxes(title_text="Time", row=3, col=1)
    fig.update_layout(
        height=500, 
        showlegend=False, 
        margin=dict(l=40, r=20, t=60, b=40),
        font=dict(size=10)
    )
    return fig

def create_weather_cards_mobile(weather_data: Dict):
    """Mobile-optimized weather cards - stacked vertically"""
    metrics = [
        ("🌡️ Temperature", f"{weather_data['temperature_2m']:.1f}°C"),
        ("💧 Humidity", f"{weather_data['relativehumidity_2m']:.1f}%"),
        ("🌧️ Precipitation", f"{weather_data['precipitation']:.1f}mm"),
        ("🤒 Feels Like", f"{weather_data['apparent_temperature']:.1f}°C"),
        ("💨 Wind Speed", f"{weather_data['windspeed_10m']:.1f}km/h")
    ]
    
    # Display 2 columns on mobile for better use of space
    cols = st.columns(2)
    for idx, (label, value) in enumerate(metrics):
        with cols[idx % 2]:
            st.metric(label, value)

def fetch_city_suspension_data(city_name: str, selected_date: datetime, df: pd.DataFrame, artifacts: Dict) -> Optional[Dict]:
    """Thread-safe function to fetch suspension data for a single city."""
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
    """Fetch suspension data for all cities in parallel using ThreadPoolExecutor."""
    all_cities_data = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_city = {
            executor.submit(fetch_city_suspension_data, city_name, selected_date, df, artifacts): city_name
            for city_name in CITY_COORDINATES.keys()
        }
        
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
# MOBILE-RESPONSIVE PAGES
# ============================================================================

def page_home():
    create_responsive_header("🌤️ HERALD v2.0", "2rem", "3.5rem")
    
    df = load_historical_data()
    artifacts = load_model_artifacts()
    min_date, max_date = get_date_bounds()
    
    with st.sidebar:
        st.markdown("### 📍 Selection Settings")
        selected_date = st.date_input(
            "Select Date", 
            value=datetime.now().date(), 
            min_value=min_date, 
            max_value=max_date, 
            key='home_date_input'
        )
        
        city = st.selectbox(
            "Select City", 
            sorted(CITY_COORDINATES.keys()), 
            index=sorted(CITY_COORDINATES.keys()).index('Manila'), 
            key='home_city'
        )
        
        mode = get_data_mode(
            datetime.combine(selected_date, datetime.min.time()), 
            df['date'].min(), 
            df['date'].max()
        )
        if mode == 'historical':
            st.success("📊 Historical Mode")
        else:
            st.info("🔮 Forecast Mode")
        
        if st.button("📅 Jump to Today", use_container_width=True):
            st.session_state.home_selected_date = datetime.now().date()
            st.rerun()
    
    # Location banner
    st.markdown(
        f"<div style='padding: 12px; background: linear-gradient(135deg, #009cdf 0%, #0066cc 100%); "
        f"border-radius: 10px; margin-bottom: 20px;'>"
        f"<h3 style='margin: 0; color: white; font-size: clamp(1.2rem, 3vw, 1.8rem);'>"
        f"📍 {city} | 📅 {selected_date.strftime('%B %d, %Y')}</h3></div>", 
        unsafe_allow_html=True
    )
    
    data = get_weather_and_suspension(
        datetime.combine(selected_date, datetime.min.time()), 
        city, df, artifacts
    )
    
    if not data:
        st.error("❌ No data available for selected date and city.")
        return
    
    # Suspension info section - mobile optimized
    level = data.get('predicted_level', data.get('suspension_level', 0))
    is_pred = data.get('is_prediction', False)
    conf = data.get('confidence')
    info = SUSPENSION_LEVELS[level]
    
    # Use expander for gauge on mobile to save space
    with st.expander("🎯 Suspension Level Details", expanded=True):
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            fig = create_suspension_gauge(level, is_pred, conf)
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
            if is_pred:
                st.metric("Confidence", f"{conf*100:.1f}%")
                st.caption("ML Prediction")
            else:
                st.caption("Historical Record")
        
        with col2:
            st.markdown(
                f"<div style='padding: 15px; background-color: {info['color']}20; "
                f"border-left: 4px solid {info['color']}; border-radius: 5px;'>"
                f"<h2 style='margin:0; font-size: clamp(1.2rem, 4vw, 2rem); color: {info['color']};'>"
                f"{info['emoji']} {info['name']}</h2>"
                f"<p style='margin: 8px 0 0 0; font-size: clamp(0.9rem, 2vw, 1.2rem);'>"
                f"<strong>Risk:</strong> {info['risk']}</p></div>", 
                unsafe_allow_html=True
            )
    
    # What This Means section
    st.markdown("### 📋 What This Means")
    descriptions = {
        0: "Normal school and work operations. Weather conditions are safe.",
        1: "Only preschool classes are suspended due to weather conditions.",
        2: "Preschool and elementary classes are suspended.",
        3: "All school levels (public schools) are suspended.",
        4: "All school levels and government work are suspended."
    }
    st.info(descriptions[level])
    
    # Weather metrics section
    st.markdown("---")
    st.markdown(f"### 🌦️ Weather Conditions")
    create_weather_cards_mobile(data['weather_data'])
    
    # Map section
    st.markdown("---")
    st.markdown("### 🗺️ Metro Manila Overview")
    
    with st.spinner('Loading map data...'):
        map_df = fetch_all_cities_parallel(selected_date, df, artifacts, max_workers=8)

    if len(map_df) == 0:
        st.warning("Unable to fetch suspension data for cities.")
    else:
        fig = go.Figure()
        
        level_colors = {
            0: '#22C55E', 1: '#FBBF24', 2: '#F97316', 3: '#EF4444', 4: '#7C2D12',
        }
        
        for lvl in range(5):
            level_data = map_df[map_df['suspension_level'] == lvl]
            if len(level_data) > 0:
                fig.add_trace(go.Scattermapbox(
                    lat=level_data['lat'], 
                    lon=level_data['lon'], 
                    mode='markers', 
                    marker=dict(size=15, color=level_colors[lvl], opacity=0.9),
                    text=level_data['city'],
                    customdata=level_data[['name']], 
                    hovertemplate='<b>%{text}</b><br>%{customdata[0]}<extra></extra>', 
                    name=SUSPENSION_LEVELS[lvl]['name'],
                    legendgroup=f'level{lvl}'
                ))
        
        selected_city_data = map_df[map_df['city'] == city]
        if len(selected_city_data) > 0:
            fig.add_trace(go.Scattermapbox(
                lat=selected_city_data['lat'], 
                lon=selected_city_data['lon'], 
                mode='markers', 
                marker=dict(size=20, color='black', symbol='star'),
                hoverinfo='skip', 
                showlegend=False
            ))
        
        fig.update_layout(
            mapbox=dict(
                style='open-street-map', 
                center=dict(lat=14.58, lon=121.0223), 
                zoom=9.5
            ), 
            height=400, 
            margin=dict(l=0, r=0, t=0, b=0), 
            showlegend=True, 
            legend=dict(
                yanchor="bottom", 
                y=0.01, 
                xanchor="left", 
                x=0.01, 
                bgcolor="rgba(255,255,255,0.9)", 
                bordercolor="black", 
                borderwidth=1,
                font=dict(size=10)
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ============================================================================
# PAGE: WEATHER ANALYTICS (Mobile Optimized)
# ============================================================================

def page_weather_analytics():
    create_responsive_header("🌦️ Weather Analytics", "1.8rem", "3rem")
    
    df = load_historical_data()
    min_date, max_date = get_date_bounds()
    
    with st.sidebar:
        st.markdown("### 📍 Selection Settings")
        selected_date = st.date_input(
            "Select Date", 
            value=datetime.now().date(), 
            min_value=min_date, 
            max_value=max_date, 
            key='analytics_date_input'
        )
        
        city = st.selectbox(
            "Select City", 
            sorted(CITY_COORDINATES.keys()), 
            index=sorted(CITY_COORDINATES.keys()).index('Manila'), 
            key='analytics_city'
        )
        
        mode = get_data_mode(
            datetime.combine(selected_date, datetime.min.time()), 
            df['date'].min(), 
            df['date'].max()
        )
        if mode == 'historical':
            st.success("📊 Historical Mode")
        else:
            st.info("🔮 Forecast Mode")
        
        if st.button("📅 Jump to Today", use_container_width=True, key='analytics_today_btn'):
            st.session_state.analytics_selected_date = datetime.now().date()
            st.rerun()
    
    data = get_weather_and_suspension(
        datetime.combine(selected_date, datetime.min.time()), 
        city, df, load_model_artifacts()
    )
    
    if not data:
        st.warning("No data available for the selected date and city.")
        return
    
    # Daily Summary
    st.markdown("### 📊 Daily Summary")
    create_weather_cards_mobile(data['weather_data'])
    
    # Suspension Status in expandable section
    with st.expander("🎯 Suspension Status", expanded=True):
        level = data.get('predicted_level', data.get('suspension_level', 0))
        is_pred = data.get('is_prediction', False)
        conf = data.get('confidence')
        
        fig_gauge = create_suspension_gauge(level, is_pred, conf)
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})
        
        if is_pred and conf is not None:
            st.metric("Confidence", f"{conf*100:.1f}%")
    
    # Hourly Trends
    st.markdown("---")
    st.markdown("### 📈 Hourly Weather Trends")
    
    hourly = data['hourly_data']
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=hourly['date'],
        y=hourly['temperature_2m'],
        name="Temperature",
        mode='lines',
        line=dict(color='red', width=2),
        yaxis='y1'
    ))
    
    fig.add_trace(go.Bar(
        x=hourly['date'],
        y=hourly['precipitation'],
        name="Precipitation",
        marker_color='dodgerblue',
        yaxis='y2',
        opacity=0.6
    ))
    
    fig.add_trace(go.Scatter(
        x=hourly['date'],
        y=hourly['windspeed_10m'],
        name="Wind Speed",
        mode='lines',
        line=dict(color='green', width=2, dash='dash'),
        yaxis='y2'
    ))
    
    fig.update_layout(
        xaxis=dict(title="Time", tickformat='%H:%M'),
        yaxis=dict(title="Temp (°C)", color='red'),
        yaxis2=dict(title="Precip/Wind", overlaying="y", side="right", color='dodgerblue'),
        legend=dict(orientation='h', yanchor='top', y=-0.15, xanchor='center', x=0.5),
        height=400,
        margin=dict(l=40, r=40, t=20, b=60),
        font=dict(size=10)
    )
    
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Weather Correlation
    st.markdown("---")
    st.markdown("### 🔗 Variable Correlation")
    
    numeric_cols = [
        'temperature_2m', 'relativehumidity_2m', 'precipitation', 
        'windspeed_10m', 'apparent_temperature'
    ]
    
    selected_var1 = st.selectbox("Variable 1", numeric_cols, key="var1")
    selected_var2 = st.selectbox("Variable 2", numeric_cols, key="var2", index=2)
    
    if selected_var1 == selected_var2:
        st.error("Please select two different variables.")
    else:
        corr_matrix = df[numeric_cols].corr()
        corr_value = corr_matrix.loc[selected_var1, selected_var2]
        abs_corr = abs(corr_value)
        
        if abs_corr < 0.2:
            label = "No correlation"
        elif abs_corr < 0.4:
            label = "Weak correlation"
        elif abs_corr < 0.7:
            label = "Moderate correlation"
        else:
            label = "Strong correlation"
        
        st.metric("Correlation", f"{corr_value:.2f}", label)
        
        # Scatter plot
        numpoints = min(len(df), 500)
        scatter_df = df[[selected_var1, selected_var2]].sample(numpoints)
        
        fig_scatter = go.Figure()
        fig_scatter.add_trace(go.Scattergl(
            x=scatter_df[selected_var1],
            y=scatter_df[selected_var2],
            mode='markers',
            marker=dict(size=5, opacity=0.6)
        ))
        
        fig_scatter.update_layout(
            xaxis_title=selected_var1.replace("_", " ").title(),
            yaxis_title=selected_var2.replace("_", " ").title(),
            height=350,
            margin=dict(l=40, r=20, t=20, b=40),
            font=dict(size=10)
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True, config={'displayModeBar': False})

# ============================================================================
# PAGE: HISTORICAL ANALYTICS (Mobile Optimized)
# ============================================================================

def page_historical_analytics():
    create_responsive_header("⌛ Historical Analytics", "1.8rem", "3rem")
    
    df = load_historical_data()
    min_date, max_date = get_date_bounds()
    
    with st.sidebar:
        st.markdown("### 📍 Date Range")
        
        if 'hist_start_date' not in st.session_state:
            st.session_state.hist_start_date = datetime(2025, 7, 20).date()
        if 'hist_end_date' not in st.session_state:
            st.session_state.hist_end_date = datetime(2025, 9, 12).date()
        
        start_date = st.date_input(
            "Start Date", 
            value=st.session_state.hist_start_date,
            min_value=min_date, 
            max_value=st.session_state.hist_end_date,
            key='hist_start'
        )
        st.session_state.hist_start_date = start_date
        
        end_date = st.date_input(
            "End Date", 
            value=st.session_state.hist_end_date,
            min_value=st.session_state.hist_start_date,
            max_value=max_date,
            key='hist_end'
        )
        st.session_state.hist_end_date = end_date
        
        selected_cities = st.multiselect(
            "Cities", 
            sorted(CITY_COORDINATES.keys()), 
            default=['Manila', 'Quezon City'], 
            key='hist_cities'
        )
    
    mask = (df['date'].dt.date >= start_date) & (df['date'].dt.date <= end_date)
    if selected_cities:
        mask = mask & (df['city'].isin(selected_cities))
    filtered_df = df[mask]
    
    if len(filtered_df) == 0:
        st.warning("No data available for selected filters.")
        return
    
    # Summary Statistics - mobile optimized
    st.markdown("### 📊 Summary Statistics")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", f"{len(filtered_df):,}")
        st.metric("Avg Precipitation", f"{filtered_df['precipitation'].mean():.1f}mm")
    with col2:
        st.metric("Suspension Rate", f"{(filtered_df['suspension'] > 0).mean() * 100:.1f}%")
        st.metric("Avg Temperature", f"{filtered_df['temperature_2m'].mean():.1f}°C")
    
    # Suspension Distribution
    st.markdown("---")
    st.markdown("### 📊 Suspension Levels")
    
    suspension_counts = filtered_df.groupby('suspension')['date'].apply(
        lambda x: x.dt.date.nunique()
    ).reset_index()
    suspension_counts.columns = ['suspension', 'days']
    suspension_counts = suspension_counts[suspension_counts['suspension'] < 5].sort_values('suspension')
    
    fig = go.Figure(go.Bar(
        x=[f"L{int(i)}" for i in suspension_counts['suspension']], 
        y=suspension_counts['days'], 
        marker_color=[SUSPENSION_LEVELS[int(i)]['color'] for i in suspension_counts['suspension']], 
        text=suspension_counts['days'], 
        textposition='auto'
    ))
    fig.update_layout(
        xaxis_title="Level", 
        yaxis_title="Days", 
        height=300,
        margin=dict(l=40, r=20, t=20, b=40),
        font=dict(size=10)
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # City Analysis
    st.markdown("---")
    st.markdown("### 🏙️ City Analysis")
    
    city_suspension = filtered_df.groupby('city')['suspension'].agg(['mean', 'count']).reset_index()
    city_suspension = city_suspension.sort_values('mean', ascending=False)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=city_suspension['city'], 
        y=city_suspension['mean'], 
        name='Avg Level',
        marker_color='#3b82f6'
    ))
    
    fig.update_layout(
        xaxis_title="City", 
        yaxis_title="Avg Suspension", 
        height=350,
        margin=dict(l=40, r=20, t=20, b=80),
        xaxis={'tickangle': -45},
        font=dict(size=9)
    )
    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Weather-Suspension Correlation
    st.markdown("---")
    st.markdown("### 🌦️ Weather Impact")
    
    # Use tabs for mobile-friendly navigation
    tab1, tab2 = st.tabs(["Precipitation", "Wind Speed"])
    
    with tab1:
        fig = px.box(
            filtered_df, 
            x='suspension', 
            y='precipitation', 
            color='suspension',
            color_discrete_map={i: SUSPENSION_LEVELS[i]['color'] for i in range(5)}
        )
        fig.update_layout(
            height=300, 
            showlegend=False,
            margin=dict(l=40, r=20, t=20, b=40),
            font=dict(size=10)
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    with tab2:
        fig = px.box(
            filtered_df, 
            x='suspension', 
            y='windspeed_10m',
            color='suspension',
            color_discrete_map={i: SUSPENSION_LEVELS[i]['color'] for i in range(5)}
        )
        fig.update_layout(
            height=300, 
            showlegend=False,
            margin=dict(l=40, r=20, t=20, b=40),
            font=dict(size=10)
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ============================================================================
# PAGE: WHAT-IF SCENARIOS (Mobile Optimized)
# ============================================================================

def page_what_if():
    create_responsive_header("🤔 What-If Scenarios", "1.8rem", "3rem")
    st.markdown("Adjust parameters to predict suspension levels")
    
    artifacts = load_model_artifacts()
    
    with st.sidebar:
        st.markdown("### 🎛️ Parameters")
        city = st.selectbox("City", sorted(CITY_COORDINATES.keys()), key='what_if_city')
        
        with st.expander("🌡️ Weather", expanded=True):
            temperature = st.slider("Temperature (°C)", 20.0, 55.0, 28.0, 0.5)
            humidity = st.slider("Humidity (%)", 30, 100, 75, 1)
            precipitation = st.slider("Precipitation (mm)", 0.0, 100.0, 10.0, 1.0)
            wind_speed = st.slider("Wind Speed (km/h)", 0, 300, 20, 1)
        
        with st.expander("⏰ Time Context"):
            hour = st.slider("Hour", 0, 23, 12)
            day_of_week = st.selectbox(
                "Day", 
                ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            )
            month = st.selectbox("Month", list(range(1, 13)), index=6)
    
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
    
    with st.spinner("Analyzing..."):
        ml_prediction = predict_suspension(weather_data, city, hourly_df, artifacts)
        pagasa_level, pagasa_reasons = check_pagasa_criteria(weather_data)
        
        final_level = max(ml_prediction['predicted_level'], pagasa_level)
        info = SUSPENSION_LEVELS[final_level]
        
        # Results Section
        st.markdown("### 🎯 Prediction Results")
        
        # Gauge
        fig = create_suspension_gauge(final_level, True, ml_prediction['confidence'])
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
        # Status card
        st.markdown(
            f"<div style='padding: 20px; background-color: {info['color']}20; "
            f"border-left: 4px solid {info['color']}; border-radius: 5px; margin: 15px 0;'>"
            f"<h2 style='margin:0; font-size: 1.5rem; color: {info['color']};'>"
            f"{info['emoji']} {info['name']}</h2>"
            f"<p style='margin: 8px 0 0 0;'><strong>Risk:</strong> {info['risk']}</p></div>", 
            unsafe_allow_html=True
        )
        
        # Justification
        with st.expander("📋 Justification", expanded=True):
            st.markdown(f"**ML Model:** Level {ml_prediction['predicted_level']} "
                       f"({ml_prediction['confidence']*100:.1f}% confidence)")
            
            if pagasa_level > 0:
                st.markdown(f"**PAGASA Criteria:** Level {pagasa_level}")
                for reason in pagasa_reasons:
                    st.markdown(f"• {reason}")
            else:
                st.markdown("**PAGASA Criteria:** No suspension")
        
        # Probabilities
        st.markdown("---")
        st.markdown("### 📊 Probability Distribution")
        
        probs = ml_prediction['probabilities']
        if len(probs) < 5:
            probs = list(probs) + [0.0] * (5 - len(probs))
        else:
            probs = list(probs)[:5]
        
        prob_df = pd.DataFrame({
            'Level': [f"L{i}" for i in range(5)], 
            'Probability': [p * 100 for p in probs]
        })
        
        fig = go.Figure(go.Bar(
            x=prob_df['Level'], 
            y=prob_df['Probability'], 
            marker_color=[SUSPENSION_LEVELS[i]['color'] for i in range(len(prob_df))], 
            text=[f"{p:.1f}%" for p in prob_df['Probability']], 
            textposition='auto'
        ))
        fig.update_layout(
            xaxis_title="Level", 
            yaxis_title="Probability (%)", 
            height=250,
            margin=dict(l=40, r=20, t=20, b=40),
            font=dict(size=10)
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

# ============================================================================
# PAGE: ABOUT (Mobile Optimized)
# ============================================================================

def page_about():
    create_responsive_header("⚙️ About", "1.8rem", "3rem")
    
    artifacts = load_model_artifacts()
    
    st.markdown("### Metro Manila Suspension Prediction System")
    st.markdown(
        "*HERALD v2.0* predicts class and work suspension levels based on "
        "weather conditions and PAGASA criteria."
    )
    
    # How It Works
    with st.expander("🔍 How It Works", expanded=False):
        st.markdown("""
        **Historical Mode:** Displays actual recorded weather and suspension decisions.
        
        **Forecast Mode:** Fetches weather forecasts and uses ML to predict suspensions.
        
        **What-If Scenarios:** Explore how different conditions affect predictions.
        """)
    
    # Model Performance
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    
    metrics = artifacts['metrics']['models']['lightgbm_tuned']
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Model", "LightGBM")
        st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
    with col2:
        balanced_acc = metrics.get('balanced_accuracy', metrics['accuracy'])
        st.metric("Balanced Accuracy", f"{balanced_acc*100:.2f}%")
        st.metric("Records", f"{artifacts['metrics']['dataset_info']['total_records']:,}")
    
    # Feature Importance
    with st.expander("🎯 Top Features", expanded=False):
        feature_imp = artifacts['feature_importance']
        top_features = dict(list(feature_imp.items())[:10])
        
        fig = go.Figure(go.Bar(
            x=list(top_features.values()), 
            y=list(top_features.keys()), 
            orientation='h', 
            marker_color='#009cdf'
        ))
        fig.update_layout(
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=350,
            margin=dict(l=100, r=20, t=20, b=40),
            yaxis={'categoryorder': 'total ascending'},
            font=dict(size=9)
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
    
    # Suspension Levels
    st.markdown("---")
    st.markdown("### 📋 Suspension Levels")
    
    for level, info in SUSPENSION_LEVELS.items():
        st.markdown(
            f"**{info['emoji']} Level {level}:** {info['name']} | Risk: {info['risk']}"
        )
    
    # Team
    st.markdown("---")
    st.markdown("### 👥 Development Team")
    
    team_members = [
        {
            "name": "Gabrielle B. Cabanilla",
            "school": "PUP",
            "contact": "cabanillla.gb@gmail.com",
            "image": "assets/gab.jpg"
        },
        {
            "name": "Jeremy Charles B. Mora",
            "school": "TIP",
            "contact": "jeremycharlesmora@gmail.com",
            "image": "assets/morsi.jpg"
        },
        {
            "name": "John Trixie M. Ocampo",
            "school": "PLM",
            "contact": "jtocampo0118@gmail.com",
            "image": "assets/jtrixie.jpg"
        }
    ]
    
    for member in team_members:
        with st.expander(f"{member['name']} - {member['school']}", expanded=False):
            if os.path.exists(member['image']):
                st.image(member['image'], width=150)
            st.markdown(f"📧 {member['contact']}")
    
    # Disclaimer
    st.markdown("---")
    st.info(
        "⚠️ This is a prediction tool. Always refer to official announcements "
        "from DepEd, PAGASA, and local government units."
    )
    
    st.markdown(
        "<div style='text-align: center; font-size: 0.8rem; color: #666; margin-top: 20px;'>"
        "<strong>HERALD v2.0</strong><br>"
        "Powered by LightGBM ML + PAGASA Criteria<br>"
        "Weather data from Open-Meteo API"
        "</div>",
        unsafe_allow_html=True
    )

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    # Inject responsive CSS
    inject_responsive_css()
    
    try:
        if not os.path.exists('model.pkl'):
            st.error("❌ Model not found! Please run 'python train_model.py' first.")
            st.code("python train_model.py", language="bash")
            return
        
        pages = {
            "🏠 Home": page_home,
            "🌦️ Weather": page_weather_analytics,
            "⌛ Historical": page_historical_analytics,
            "🤔 What-If": page_what_if,
            "⚙️ About": page_about
        }

        with st.sidebar:
            st.markdown("---")
            # Mobile-friendly navigation
            if 'selected_page' not in st.session_state:
                st.session_state.selected_page = "🏠 Home"
            
            st.markdown("### 📍 Navigation")
            
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
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666; padding: 15px; font-size: 0.85rem;'>"
            "<strong>HERALD v2.0</strong><br>"
            "Hydrometeorological Early Risk Assessment<br>"
            "Powered by LightGBM ML + PAGASA Criteria"
            "</div>", 
            unsafe_allow_html=True
        )
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.exception(e)

if __name__ == "__main__":
    main()