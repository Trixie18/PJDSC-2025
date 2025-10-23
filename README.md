# 🌤️ HERALD v2.0

**Hydrometeorological Early Risk Assessment and Live Decision-support**

A machine learning-powered suspension prediction system for Metro Manila that forecasts class and work suspension levels based on weather conditions and official PAGASA meteorological criteria.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg)](https://streamlit.io/)
[![LightGBM](https://img.shields.io/badge/LightGBM-Latest-lightgrey.svg)](https://lightgbm.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table of Contents

- [Overview](#Overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Contributing](#contributing)
- [Team](#team)
- [License](#license)

---

## 🎯 Overview

HERALD v2.0 is an intelligent decision-support system designed to predict class and work suspension levels across Metro Manila's 17 cities. By combining machine learning predictions with official PAGASA criteria, the system provides accurate, real-time forecasts to help institutions make informed decisions about suspensions.

### Key Capabilities

- **Historical Analysis**: View actual suspension decisions and weather patterns from 2020-2025
- **Real-time Forecasting**: Predict suspensions up to 7 days ahead using Open-Meteo API
- **What-If Scenarios**: Simulate different weather conditions to understand suspension likelihood
- **Multi-City Coverage**: Supports all 17 cities in Metro Manila
- **PAGASA Compliance**: Validates predictions against official meteorological criteria
- **PDF Reports**: Generate professional memorandums with weather analytics

---

## ✨ Features

### 🏠 Home Dashboard
- Real-time suspension level predictions
- Interactive Metro Manila map with color-coded suspension levels
- Comprehensive weather metrics display
- Historical vs. forecast data modes

### 🌦️ Weather Analytics
- Hourly weather trend visualization (temperature, precipitation, wind)
- Weather variable correlation analysis
- Suspension gauge with confidence scores
- City-by-city suspension mapping

### ⌛ Historical Analytics
- Customizable date range analysis
- Multi-city comparison
- Suspension level distribution charts
- Weather-suspension correlation plots
- Temporal trend analysis
- Hour vs. Day of Week heatmaps

### 🤔 What-If Scenario Predictor
- Manual weather parameter adjustment
- Real-time prediction updates
- Probability distribution visualization
- ML model + PAGASA criteria comparison

### 📄 PDF Memorandum Generation
- Professional suspension memorandum format
- Weather data summaries
- ML prediction confidence breakdown
- PAGASA criteria assessment
- Comprehensive weather analytics annex

---

## 🏗️ System Architecture

### Technology Stack

**Frontend**
- Streamlit (Interactive web interface)
- Plotly (Dynamic visualizations)
- HTML/CSS (Custom styling)

**Backend**
- Python 3.8+
- LightGBM (ML classifier)
- scikit-learn (Preprocessing & evaluation)
- pandas/numpy (Data manipulation)

**Data Sources**
- Historical: Custom Metro Manila dataset (2020-2025)
- Forecast: Open-Meteo API
- Validation: PAGASA suspension criteria

**Deployment**
- Streamlit Cloud compatible
- Docker ready
- Local development support

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- Internet connection (for forecast mode)

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/Trixie18/PJDSC-2025.git
cd PJDSC-2025
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Verify dataset**
Ensure `metro_manila_weather_sus_data.csv` is in the project root.

4. **Train the model** (first time only)
```bash
python train_model.py
```
This will generate:
- `model.pkl` - Trained LightGBM classifier
- `preprocessor.pkl` - Feature scaler
- `city_encoder.pkl` - City label encoder
- `label_encoder.pkl` - Target encoder
- `metrics.json` - Performance metrics
- `feature_importance.json` - Feature rankings
- `thresholds.json` - Suspension level statistics

5. **Launch the application**
```bash
streamlit run main.py
```

6. **Access the app**
Open your browser to `http://localhost:8501`

### Dependencies

Create `requirements.txt` with:

```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
plotly>=5.17.0
requests>=2.31.0
joblib>=1.3.0
fpdf>=1.7.2
playwright>=1.40.0
```

For PDF export functionality:
```bash
playwright install chromium
```

---

## 💻 Usage

### Running the Application

**Development Mode**
```bash
streamlit run main.py --server.runOnSave true
```

**Production Mode**
```bash
streamlit run main.py --server.port 8501 --server.address 0.0.0.0
```

### Navigation Guide

#### 1. Home Page
- Select date and city from sidebar
- View suspension level gauge
- Check weather metrics
- Explore Metro Manila suspension map
- Download PDF memorandum

#### 2. Weather Analytics
- Analyze hourly weather trends
- Compare weather variables
- View correlation scatter plots
- Generate detailed reports

#### 3. Historical Analytics
- Set custom date ranges
- Select multiple cities
- Review suspension statistics
- Analyze temporal patterns
- Study weather-suspension relationships

#### 4. What-If Scenarios
- Adjust weather sliders
- Set temporal context
- Compare ML vs. PAGASA predictions
- Explore probability distributions

#### 5. About
- Review model performance
- Understand feature importance
- Check dataset information
- Learn about PAGASA criteria

---

## 🤖 Model Details

### Machine Learning Pipeline

**Algorithm**: LightGBM Classifier (Gradient Boosting)

**Training Strategy**:
1. **Stratified Split**: 80% train, 20% test with class preservation
2. **Aggressive Oversampling**: Minority classes oversampled to 80% of majority
3. **Class Weighting**: Inverse frequency weights for loss balancing
4. **Noise Injection**: 2% Gaussian noise to prevent overfitting

**Hyperparameters** (Optimized):
```python
{
    'learning_rate': 0.03,
    'n_estimators': 800,
    'max_depth': 10,
    'num_leaves': 100,
    'subsample': 0.9,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.3,
    'reg_lambda': 2.0,
    'min_child_samples': 10,
    'boost_from_average': False
}
```

### Feature Engineering

**Base Features** (5):
- Temperature, Humidity, Precipitation, Apparent Temperature, Wind Speed

**Temporal Features** (5):
- Hour, Day of Week, Month, Weekend Flag, Rush Hour Flag

**Rolling Features** (6):
- 3h, 6h, 12h averages for precipitation and wind

**Lag Features** (6):
- 1h, 2h, 3h lags for precipitation and temperature

**Derived Features** (5):
- Precipitation peaks, Wind peaks, Temperature delta, Precipitation intensity, Wind category

**Total**: 28 features

### Performance Metrics

| Metric | Score |
|--------|-------|
| Weighted Accuracy | ~99% |
| Balanced Accuracy | Optimized for imbalance |
| Macro F1 Score | Class-equitable |
| Cross-Validation | 5-fold stratified |

**Note**: Metrics prioritize balanced accuracy due to extreme class imbalance (~98% Level 0).

### PAGASA Criteria Integration

The system validates ML predictions against official criteria:

**Wind-Based (TCWS)**:
- Signal 1 (30-60 km/h) → Preschool suspension
- Signal 2 (60-100 km/h) → Preschool + Elementary
- Signal 3 (100-150 km/h) → All school levels
- Signal 4 (150+ km/h) → All levels + work

**Rainfall-Based**:
- Light (≥7.5 mm/h) → Preschool suspension
- Moderate (≥15 mm/h) → All school levels
- Heavy (≥30 mm/h) → All levels + work

**Heat Index-Based**:
- 41°C → All school levels suspended
- 51°C → All levels + work suspended

**Final Decision**: `max(ML_prediction, PAGASA_level)`

---

## 📊 Dataset

### Data Sources

**Cities Covered** (17):
Caloocan, Las Piñas, Makati, Malabon, Mandaluyong, Manila, Marikina, Muntinlupa, Navotas, Parañaque, Pasay, Pasig, Pateros, Quezon City, San Juan, Taguig, Valenzuela

**Date Range**: September 20, 2020 - October 10, 2025

**Weather Variables**:
- Temperature (°C)
- Relative Humidity (%)
- Precipitation (mm)
- Apparent Temperature (°C)
- Wind Speed (km/h)

**Suspension Levels**:
- **Level 0**: No suspension
- **Level 1**: Preschool only
- **Level 2**: Preschool + Elementary
- **Level 3**: All levels (public schools)
- **Level 4**: All levels + work

**Data Collection**:
- Historical weather: Open-Meteo Historical API
- Suspension records: Official LGU announcements (Facebook pages)
- Forecast data: Open-Meteo Forecast API

### Class Distribution

| Level | Description | Approximate % |
|-------|-------------|---------------|
| 0 | No Suspension | ~98% |
| 1 | Preschool Only | ~1% |
| 2 | Preschool + Elem | <1% |
| 3 | All Levels | <1% |
| 4 | All + Work | <0.1% |

**Challenge**: Extreme imbalance handled through oversampling, class weighting, and specialized metrics.

---

## 📁 Project Structure

```
PJDSC-2025/
│
├── main.py                              # Streamlit application
├── train_model.py                       # Model training script
├── metro_manila_weather_sus_data.csv    # Dataset
│
├── assets/                              # Images for About page
│   ├── gab.jpg
│   ├── morsi.jpg
│   └── jtrixie.jpg
│
├── model.pkl                            # Trained model
├── preprocessor.pkl                     # Feature scaler
├── city_encoder.pkl                     # City encoder
├── label_encoder.pkl                    # Target encoder
├── metrics.json                         # Evaluation metrics
├── feature_importance.json              # Feature rankings
├── thresholds.json                      # Suspension thresholds
├── confusion_matrix.npy                 # Confusion matrix
├── training_stats.json                  # Training summary
├── model_ready.flag                     # Deployment flag
│
├── requirements.txt                     # Python dependencies
├── README.md                            # This file
└── LICENSE                              # License information
```

---

## 🔌 API Reference

### Key Functions

#### `get_weather_and_suspension(selected_date, city, df, artifacts)`
Fetches weather data and predicts/retrieves suspension level.

**Parameters**:
- `selected_date` (datetime): Target date
- `city` (str): City name
- `df` (DataFrame): Historical dataset
- `artifacts` (dict): Model artifacts

**Returns**: Dictionary with weather data, suspension level, and metadata

#### `predict_suspension(weather_data, city, hourly_df, artifacts)`
Generates ML prediction for given weather conditions.

**Parameters**:
- `weather_data` (dict): Weather metrics
- `city` (str): City name
- `hourly_df` (DataFrame): Hourly weather data
- `artifacts` (dict): Model artifacts

**Returns**: Dictionary with prediction, probabilities, confidence, risk level

#### `check_pagasa_criteria(weather_data)`
Validates weather against PAGASA suspension criteria.

**Parameters**:
- `weather_data` (dict): Weather metrics

**Returns**: Tuple of (suspension_level, reasons_list)

#### `generate_suspension_memorandum_pdf(...)`
Creates official PDF memorandum.

**Parameters**: Various (see docstring in code)

**Returns**: Path to generated PDF file

---

## 🤝 Contributing

We welcome contributions! Here's how to get started:

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Test thoroughly
5. Commit: `git commit -m 'Add amazing feature'`
6. Push: `git push origin feature/amazing-feature`
7. Open a Pull Request

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include unit tests for new features
- Update README for significant changes
- Test on multiple cities and date ranges

### Areas for Contribution

- [ ] Additional weather data sources
- [ ] More sophisticated ML models
- [ ] Mobile-responsive UI improvements
- [ ] Real-time notification system
- [ ] Historical data expansion
- [ ] Multi-language support
- [ ] API endpoint creation

---

## 👥 Team

### Development Team

**Gabrielle B. Cabanilla**
- School: Polytechnic University of the Philippines
- Role: Railway Engineering Graduate
- Email: cabanillla.gb@gmail.com

**Jeremy Charles B. Mora**
- School: Technological Institute of the Philippines
- Role: Computer Science Major - Data Analyst
- Email: jeremycharlesmora@gmail.com

**John Trixie M. Ocampo**
- School: Pamantasan ng Lungsod ng Maynila
- Role: BS Mathematics Graduate
- Email: jtocampo0118@gmail.com

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

HERALD v2.0 is a prediction tool and should **NOT** be used as the sole basis for official suspension decisions. Always refer to official announcements from:

- Department of Education (DepEd)
- DOST-PAGASA
- Local Government Units (LGUs)

This system is designed as an **auxiliary decision-support tool** only.

---

## 📞 Support

For questions, issues, or suggestions:

- **GitHub Issues**: [Create an issue](https://github.com/Trixie18/PJDSC-2025/issues)
- **Email**: Contact any team member above
- **Competition**: Philippine Junior Data Science Challenge 2025

---

## 🙏 Acknowledgments

- **Open-Meteo**: Weather API provider
- **PAGASA**: Official meteorological guidelines
- **Metro Manila LGUs**: Historical suspension data
- **PJDSC 2025**: Competition organizers
- **LightGBM Team**: ML framework
- **Streamlit**: Web framework

---

## 📈 Version History

### v2.0 (October 2025)
- Enhanced ML model with aggressive class balancing
- PAGASA criteria integration
- Professional PDF memorandum generation
- Multi-city parallel data fetching
- Improved visualization suite
- Comprehensive analytics dashboard

### v1.0 (Initial Release)
- Basic suspension prediction
- Historical data analysis
- Simple web interface

---

## 🎯 Future Roadmap

- [ ] Real-time alert system via SMS/Email
- [ ] Mobile application (iOS/Android)
- [ ] Ensemble model with multiple algorithms
- [ ] Integration with official DepEd systems
- [ ] Expanded coverage beyond Metro Manila
- [ ] Weather satellite imagery integration
- [ ] Social media sentiment analysis
- [ ] Multi-year seasonal pattern analysis

---

**Built with ❤️ for safer, more informed suspension decisions in Metro Manila**

**Version**: 2.0 | **Last Updated**: October 2025 | **Powered by**: LightGBM + PAGASA Criteria
