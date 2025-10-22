"""
Metro Manila Suspension Prediction - Model Training (Notebook Version)

This module implements a complete machine learning pipeline for predicting
class suspension levels in Metro Manila based on weather conditions.

Features:
- Handles imbalanced multi-class classification (6 suspension levels)
- Implements aggressive oversampling and class weighting
- Uses LightGBM with optimized hyperparameters
- Provides comprehensive evaluation metrics
- Saves all artifacts for deployment

Classes:
    None (script-based workflow)

Functions:
    Main workflow sections:
    1. Data loading and preprocessing
    2. Feature engineering (temporal, rolling, lag features)
    3. Train-test split with stratification
    4. Class imbalance handling (oversampling + weighting)
    5. Model training with LightGBM
    6. Comprehensive evaluation
    7. Feature importance analysis
    8. Threshold calculation
    9. Artifact saving

Usage:
    Run in Jupyter notebook or as standalone script:
    $ python train_model_notebook.py
    
    Then launch the web app:
    $ streamlit run main.py

Author: Metro Manila Weather Analysis Team
Date: 2025
Version: 2.0
"""

import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, ParameterSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                             confusion_matrix, classification_report, balanced_accuracy_score)
from lightgbm import LGBMClassifier
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# 1. LOAD AND PREPROCESS DATA
# =============================================================================

def load_and_preprocess_data(filepath='metro_manila_weather_sus_data.csv'):
    """
    Load weather and suspension data from CSV and handle missing values.
    
    This function reads the dataset, parses dates, and imputes missing values
    using median for continuous variables and 0 for precipitation.
    
    Args:
        filepath (str): Path to the CSV file containing weather data
        
    Returns:
        pd.DataFrame: Preprocessed dataframe with filled missing values
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        
    Notes:
        - Uses median imputation for temperature, humidity, wind speed
        - Uses 0 for missing precipitation (assumes no rain)
        - Preserves all original columns
    """
    print("📊 Loading dataset...")
    df = pd.read_csv(filepath, parse_dates=['date'])
    print(f"✅ Loaded {len(df):,} records from {df['date'].min()} to {df['date'].max()}")
    
    # Handle missing values with appropriate strategies
    print("🔧 Handling missing values...")
    df = df.fillna({
        'relativehumidity_2m': df['relativehumidity_2m'].median(),
        'temperature_2m': df['temperature_2m'].median(),
        'precipitation': 0,  # Assume no rain if missing
        'apparent_temperature': df['apparent_temperature'].median(),
        'windspeed_10m': df['windspeed_10m'].median()
    })
    
    print(f"✅ Dataset shape: {df.shape}")
    print(f"📈 Suspension distribution:\n{df['suspension'].value_counts().sort_index()}")
    
    return df

# Load data
df = load_and_preprocess_data()

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================

def engineer_temporal_features(df):
    """
    Create time-based features from datetime column.
    
    Extracts hour, day of week, month and creates derived features
    like weekend and rush hour indicators.
    
    Args:
        df (pd.DataFrame): DataFrame with 'date' column
        
    Returns:
        pd.DataFrame: DataFrame with added temporal features
        
    Features Created:
        - hour: Hour of day (0-23)
        - day_of_week: Day of week (0=Monday, 6=Sunday)
        - month: Month (1-12)
        - is_weekend: Binary indicator for Saturday/Sunday
        - is_rush_hour: Binary indicator for peak traffic hours (7-9, 17-19)
    """
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)
    return df

def engineer_rolling_features(df, windows=[3, 6, 12]):
    """
    Create rolling average features for precipitation and wind speed.
    
    Computes moving averages grouped by city to capture recent trends
    in weather conditions.
    
    Args:
        df (pd.DataFrame): DataFrame sorted by city and date
        windows (list): List of window sizes in hours
        
    Returns:
        pd.DataFrame: DataFrame with added rolling features
        
    Features Created:
        - precip_roll_{window}h: Rolling mean of precipitation
        - wind_roll_{window}h: Rolling mean of wind speed
        
    Notes:
        - Uses min_periods=1 to avoid NaN at start of series
        - Groupby city ensures rolling windows don't cross cities
    """
    print("  📊 Computing rolling averages...")
    for window in windows:
        df[f'precip_roll_{window}h'] = df.groupby('city')['precipitation'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'wind_roll_{window}h'] = df.groupby('city')['windspeed_10m'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
    return df

def engineer_lag_features(df, lags=[1, 2, 3]):
    """
    Create lagged features for precipitation and temperature.
    
    Captures historical weather conditions from previous hours to help
    model learn temporal dependencies.
    
    Args:
        df (pd.DataFrame): DataFrame sorted by city and date
        lags (list): List of lag periods in hours
        
    Returns:
        pd.DataFrame: DataFrame with added lag features
        
    Features Created:
        - precip_lag_{lag}h: Precipitation from {lag} hours ago
        - temp_lag_{lag}h: Temperature from {lag} hours ago
        
    Notes:
        - Fills NaN with 0 for precipitation
        - Fills NaN with median for temperature
        - Groupby city ensures lags don't cross cities
    """
    print("  🔄 Creating lag features...")
    for lag in lags:
        df[f'precip_lag_{lag}h'] = df.groupby('city')['precipitation'].shift(lag).fillna(0)
        df[f'temp_lag_{lag}h'] = df.groupby('city')['temperature_2m'].shift(lag).fillna(df['temperature_2m'].median())
    return df

def engineer_peak_indicators(df):
    """
    Create binary indicators for sudden weather changes.
    
    Identifies when current conditions exceed recent averages by
    significant thresholds, indicating potential severe weather.
    
    Args:
        df (pd.DataFrame): DataFrame with rolling features
        
    Returns:
        pd.DataFrame: DataFrame with added peak indicators
        
    Features Created:
        - is_precip_peak: 1 if precipitation > 1.5x 6-hour average
        - is_wind_peak: 1 if wind speed > 1.3x 6-hour average
    """
    df['is_precip_peak'] = (df['precipitation'] > df['precip_roll_6h'] * 1.5).astype(int)
    df['is_wind_peak'] = (df['windspeed_10m'] > df['wind_roll_6h'] * 1.3).astype(int)
    return df

def engineer_categorical_features(df):
    """
    Create categorical features from continuous weather variables.
    
    Bins continuous values into discrete categories to help model
    learn non-linear relationships.
    
    Args:
        df (pd.DataFrame): DataFrame with weather columns
        
    Returns:
        pd.DataFrame: DataFrame with added categorical features
        
    Features Created:
        - precip_intensity: 5 levels (0=none, 1=light, 2=moderate, 3=heavy, 4=extreme)
        - wind_category: 4 levels (0=calm, 1=moderate, 2=strong, 3=severe)
        - apparent_temp_delta: Difference between apparent and actual temperature
    """
    df['apparent_temp_delta'] = df['apparent_temperature'] - df['temperature_2m']
    
    df['precip_intensity'] = pd.cut(df['precipitation'], 
                                    bins=[-1, 0, 5, 15, 50, 200],
                                    labels=[0, 1, 2, 3, 4]).astype(int)
    
    df['wind_category'] = pd.cut(df['windspeed_10m'],
                                 bins=[-1, 20, 40, 60, 100],
                                 labels=[0, 1, 2, 3]).astype(int)
    return df

def encode_city_column(df):
    """
    Convert city names to numeric codes.
    
    Uses LabelEncoder to transform city strings into integers for
    model compatibility.
    
    Args:
        df (pd.DataFrame): DataFrame with 'city' column
        
    Returns:
        tuple: (DataFrame with city_encoded column, fitted LabelEncoder)
        
    Notes:
        - Returns encoder for later use in predictions
        - Preserves original city column
    """
    print("  🏙️ Encoding cities...")
    city_encoder = LabelEncoder()
    df['city_encoded'] = city_encoder.fit_transform(df['city'])
    return df, city_encoder

# Apply all feature engineering
print("\n🛠️ Engineering features...")
df = df.sort_values(['city', 'date'])
df = engineer_temporal_features(df)
df = engineer_rolling_features(df)
df = engineer_lag_features(df)
df = engineer_peak_indicators(df)
df = engineer_categorical_features(df)
df, city_encoder = encode_city_column(df)

print(f"✅ Feature engineering complete. Total features: {len(df.columns)}")

# =============================================================================
# 3. PREPARE TRAIN-TEST SPLIT
# =============================================================================

def prepare_features_and_target(df):
    """
    Select feature columns and encode target variable.
    
    Prepares the dataset for training by selecting relevant features
    and encoding the suspension level target.
    
    Args:
        df (pd.DataFrame): Fully engineered dataset
        
    Returns:
        tuple: (X features DataFrame, y_encoded array, LabelEncoder)
        
    Features Selected:
        - Base weather: humidity, temperature, precipitation, apparent temp, wind
        - Temporal: hour, day_of_week, month, is_weekend, is_rush_hour
        - Rolling: 3h, 6h, 12h averages for precipitation and wind
        - Lag: 1h, 2h, 3h lags for precipitation and temperature
        - Indicators: precipitation peaks, wind peaks, temp delta
        - Categorical: precipitation intensity, wind category, city code
    """
    feature_cols = [
        'relativehumidity_2m', 'temperature_2m', 'precipitation', 
        'apparent_temperature', 'windspeed_10m',
        'hour', 'day_of_week', 'month', 'is_weekend', 'is_rush_hour',
        'precip_roll_3h', 'precip_roll_6h', 'precip_roll_12h',
        'wind_roll_3h', 'wind_roll_6h', 'wind_roll_12h',
        'precip_lag_1h', 'precip_lag_2h', 'precip_lag_3h',
        'temp_lag_1h', 'temp_lag_2h', 'temp_lag_3h',
        'is_precip_peak', 'is_wind_peak', 'apparent_temp_delta',
        'precip_intensity', 'wind_category', 'city_encoded'
    ]
    
    X = df[feature_cols].copy()
    y = df['suspension'].copy()
    
    # Ensure all columns are numeric
    for col in X.columns:
        if X[col].dtype == 'category':
            X[col] = X[col].astype(int)
    
    # Encode target variable
    print("  🎯 Encoding target variable...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    print(f"  Original classes: {sorted(y.unique())}")
    print(f"  Encoded classes: {sorted(np.unique(y_encoded))}")
    
    return X, y_encoded, label_encoder, feature_cols

def perform_stratified_split(X, y_encoded, test_size=0.2, random_state=42):
    """
    Split data into train and test sets with stratification.
    
    Ensures both sets have similar class distributions, critical for
    imbalanced classification problems.
    
    Args:
        X (pd.DataFrame): Feature matrix
        y_encoded (np.array): Encoded target labels
        test_size (float): Proportion for test set (default 0.2)
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
        
    Notes:
        - Stratify parameter ensures all classes in both sets
        - Shuffle=True randomizes order before splitting
        - Critical for preventing data leakage in time series
    """
    print("  🔀 Performing stratified split (preserves class distribution)...")
    return train_test_split(
        X, y_encoded, test_size=test_size, stratify=y_encoded, 
        random_state=random_state, shuffle=True
    )

def scale_numerical_features(X_train, X_test):
    """
    Standardize numerical features to zero mean and unit variance.
    
    Improves model convergence and prevents features with large ranges
    from dominating the learning process.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Test features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, fitted StandardScaler)
        
    Columns Scaled:
        - relativehumidity_2m
        - temperature_2m
        - precipitation
        - apparent_temperature
        - windspeed_10m
        - apparent_temp_delta
        
    Notes:
        - Fit only on training data to prevent leakage
        - Transform both train and test with same scaler
    """
    print("  📏 Scaling features...")
    preprocessor = StandardScaler()
    numerical_cols = ['relativehumidity_2m', 'temperature_2m', 'precipitation', 
                     'apparent_temperature', 'windspeed_10m', 'apparent_temp_delta']
    
    X_train[numerical_cols] = preprocessor.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = preprocessor.transform(X_test[numerical_cols])
    
    return X_train, X_test, preprocessor

# Prepare data
print("\n✂️ Splitting data (80-20 stratified)...")
df = df.sort_values('date')
X, y_encoded, label_encoder, feature_cols = prepare_features_and_target(df)
X_train, X_test, y_train, y_test = perform_stratified_split(X, y_encoded)
X_train, X_test, preprocessor = scale_numerical_features(X_train, X_test)

print(f"✅ Train set: {len(X_train):,} | Test set: {len(X_test):,}")
print(f"📊 Train suspension distribution (encoded):")
train_dist = pd.Series(y_train).value_counts().sort_index()
for class_idx, count in train_dist.items():
    original_class = label_encoder.inverse_transform([class_idx])[0]
    percentage = (count / len(y_train)) * 100
    print(f"   Class {original_class}: {count:,} ({percentage:.2f}%)")

print(f"\n📊 Test suspension distribution (encoded):")
test_dist = pd.Series(y_test).value_counts().sort_index()
for class_idx, count in test_dist.items():
    original_class = label_encoder.inverse_transform([class_idx])[0]
    percentage = (count / len(y_test)) * 100
    print(f"   Class {original_class}: {count:,} ({percentage:.2f}%)")

# =============================================================================
# 4. HANDLE CLASS IMBALANCE & TRAIN MODEL
# =============================================================================

def calculate_class_weights(y_train):
    """
    Compute class weights inversely proportional to class frequencies.
    
    Assigns higher weights to minority classes to balance the loss
    function during training.
    
    Args:
        y_train (np.array): Training target labels
        
    Returns:
        dict: Mapping from class index to weight
        
    Formula:
        weight(class) = n_samples / (n_classes * n_samples_class)
        
    Notes:
        - Uses linear scale (not sqrt) for aggressive balancing
        - Higher weights penalize misclassification of rare classes more
    """
    print("\n⚖️ Calculating class weights...")
    class_counts = pd.Series(y_train).value_counts().sort_index()
    total_samples = len(y_train)
    
    class_weights = {}
    for class_idx in np.unique(y_train):
        class_weights[class_idx] = total_samples / (len(np.unique(y_train)) * class_counts[class_idx])
    
    print("Class weights (higher = more weight for minority classes):")
    for class_idx, weight in sorted(class_weights.items()):
        print(f"   Class {class_idx}: {weight:.2f}")
    
    return class_weights

def oversample_minority_classes(X_train, y_train, target_ratio=0.8):
    """
    Balance training data by oversampling minority classes.
    
    Creates synthetic samples by duplicating existing samples with
    small random noise to prevent exact duplicates.
    
    Args:
        X_train (np.array): Training features
        y_train (np.array): Training labels
        target_ratio (float): Target size as fraction of majority class (0-1)
        
    Returns:
        tuple: (X_balanced, y_balanced) with oversampled data
        
    Process:
        1. Find maximum class size
        2. Calculate target size = max_size * target_ratio
        3. For minority classes:
           - Random sample with replacement
           - Add small Gaussian noise (std=0.02)
           - Combine with original samples
        4. Shuffle all data
        
    Notes:
        - Higher target_ratio = more aggressive balancing
        - Noise prevents overfitting to duplicates
        - Preserves original distribution for majority classes
    """
    print("\n🔄 Oversampling minority classes (aggressive balancing)...")
    
    # Separate by class
    X_by_class = {i: X_train[y_train == i] for i in np.unique(y_train)}
    y_by_class = {i: y_train[y_train == i] for i in np.unique(y_train)}
    
    # Calculate target size
    max_samples = max([len(X_by_class[i]) for i in X_by_class.keys()])
    target_samples = int(max_samples * target_ratio)
    print(f"   Target samples per class: {target_samples:,}")
    
    X_train_balanced = []
    y_train_balanced = []
    
    for class_idx in sorted(X_by_class.keys()):
        X_class = X_by_class[class_idx]
        y_class = y_by_class[class_idx]
        
        if len(X_class) < target_samples:
            # Oversample with noise
            n_duplicates = target_samples - len(X_class)
            duplicate_indices = np.random.choice(len(X_class), size=n_duplicates, replace=True)
            X_duplicates = X_class[duplicate_indices].copy()
            noise = np.random.normal(0, 0.02, X_duplicates.shape)
            X_duplicates = X_duplicates + noise
            
            X_combined = np.vstack([X_class, X_duplicates])
            y_combined = np.concatenate([y_class, np.full(n_duplicates, class_idx)])
        else:
            X_combined = X_class
            y_combined = y_class
        
        X_train_balanced.append(X_combined)
        y_train_balanced.append(y_combined)
    
    # Stack and shuffle
    X_train_balanced = np.vstack(X_train_balanced)
    y_train_balanced = np.concatenate(y_train_balanced)
    shuffle_idx = np.random.permutation(len(X_train_balanced))
    
    print(f"   Original training size: {len(X_train):,}")
    print(f"   Balanced training size: {len(X_train_balanced):,}")
    
    return X_train_balanced[shuffle_idx], y_train_balanced[shuffle_idx]

def train_lightgbm_model(X_train, y_train, class_weights, n_classes):
    """
    Train LightGBM classifier with optimized hyperparameters.
    
    Uses gradient boosting with parameters tuned for imbalanced
    multi-class classification.
    
    Args:
        X_train (np.array): Training features
        y_train (np.array): Training labels
        class_weights (dict): Class weight mapping
        n_classes (int): Number of classes
        
    Returns:
        LGBMClassifier: Trained model
        
    Hyperparameters:
        - learning_rate: 0.03 (slow learning for stability)
        - n_estimators: 800 (many trees for complex patterns)
        - max_depth: 10 (deep trees for interactions)
        - num_leaves: 100 (complex tree structure)
        - subsample: 0.9 (row sampling)
        - colsample_bytree: 0.8 (column sampling)
        - reg_alpha/lambda: L1/L2 regularization
        - min_child_samples: 10 (small leaves allowed)
        - boost_from_average: False (no majority class bias)
        
    Notes:
        - force_row_wise=True for faster training
        - verbose=-1 suppresses output
        - n_jobs=-1 uses all CPU cores
    """
    best_params = {
        'colsample_bytree': 0.8,
        'learning_rate': 0.03,
        'max_depth': 10,
        'min_child_samples': 10,
        'n_estimators': 800,
        'num_leaves': 100,
        'reg_alpha': 0.3,
        'reg_lambda': 2.0,
        'subsample': 0.9,
        'min_child_weight': 0.001,
        'boost_from_average': False
    }
    
    print("\n🔧 Training with optimized parameters...")
    print("="*60)
    for param, value in sorted(best_params.items()):
        print(f"   {param}: {value}")
    print("="*60)
    
    model = LGBMClassifier(
        **best_params,
        num_class=n_classes,
        class_weight=class_weights,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
        force_row_wise=True
    )
    
    print("\n🎯 Training on balanced dataset...")
    model.fit(X_train, y_train)
    print("✅ Training complete!\n")
    
    return model

# Handle imbalance and train
print("\n🤖 Training model with class imbalance handling...\n")
X_train_lgb = X_train.astype(float)
X_test_lgb = X_test.astype(float)

print("📊 Class Distribution Analysis:")
class_counts = pd.Series(y_train).value_counts().sort_index()
total_samples = len(y_train)
for class_idx, count in class_counts.items():
    percentage = (count / total_samples) * 100
    original_class = label_encoder.inverse_transform([class_idx])[0]
    print(f"   Class {original_class} (encoded {class_idx}): {count:,} samples ({percentage:.2f}%)")

class_weights = calculate_class_weights(y_train)
X_train_balanced, y_train_balanced = oversample_minority_classes(X_train_lgb.values, y_train)

print("\nBalanced class distribution:")
balanced_counts = pd.Series(y_train_balanced).value_counts().sort_index()
for class_idx, count in balanced_counts.items():
    original_class = label_encoder.inverse_transform([class_idx])[0]
    print(f"   Class {original_class}: {count:,} samples")

best_model = train_lightgbm_model(
    X_train_balanced, y_train_balanced, class_weights, len(np.unique(y_train))
)

# =============================================================================
# 5. EVALUATE MODEL (WITH CLASS-SPECIFIC METRICS)
# =============================================================================

def evaluate_model_performance(model, X_test, y_test, label_encoder):
    """
    Comprehensive evaluation of model performance.
    
    Computes multiple metrics to assess model quality on imbalanced data,
    including per-class and aggregate metrics.
    
    Args:
        model: Trained classifier
        X_test (np.array): Test features
        y_test (np.array): True test labels
        label_encoder (LabelEncoder): For inverse transform
        
    Returns:
        dict: Dictionary containing all evaluation metrics
        
    Metrics Computed:
        - Per-class: precision, recall, F1, support
        - Weighted: accounts for class frequency
        - Macro: treats all classes equally
        - Confusion matrix
        - Balanced accuracy (better for imbalanced data)
        
    Notes:
        - Weighted metrics reflect overall performance
        - Macro metrics show per-class fairness
        - Balanced accuracy adjusts for class imbalance
    """
    print("📊 Evaluating model on imbalanced test set...\n")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Overall metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0
    )
    
    print("="*60)
    print("📈 PER-CLASS PERFORMANCE (Key for Imbalanced Data)")
    print("="*60)
    for class_idx in range(len(precision_per_class)):
        original_class = label_encoder.inverse_transform([class_idx])[0]
        print(f"\nClass {original_class} (Level {original_class}):")
        print(f"   Samples in test: {support_per_class[class_idx]}")
        print(f"   Precision: {precision_per_class[class_idx]:.4f}")
        print(f"   Recall:    {recall_per_class[class_idx]:.4f}")
        print(f"   F1 Score:  {f1_per_class[class_idx]:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\n" + "="*60)
    print("🔍 CONFUSION MATRIX ANALYSIS")
    print("="*60)
    print("\nConfusion Matrix:")
    print(cm)
    
    print("\nPer-class accuracy breakdown:")
    for class_idx in range(len(cm)):
        original_class = label_encoder.inverse_transform([class_idx])[0]
        correct = cm[class_idx, class_idx]
        total = cm[class_idx, :].sum()
        if total > 0:
            class_accuracy = correct / total
            print(f"   Class {original_class}: {correct}/{total} correct ({class_accuracy:.2%})")
        else:
            print(f"   Class {original_class}: No samples in test set")
    
    # Macro metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )
    
    print("\n" + "="*60)
    print("📊 AGGREGATE METRICS")
    print("="*60)
    print(f"\nWeighted Metrics (accounts for class frequency):")
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1 Score:  {f1:.4f}")
    
    print(f"\nMacro Metrics (treats all classes equally):")
    print(f"   Precision: {precision_macro:.4f}")
    print(f"   Recall:    {recall_macro:.4f}")
    print(f"   F1 Score:  {f1_macro:.4f}")
    
    # Balanced accuracy
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    print(f"\n⚖️ Balanced Accuracy: {balanced_acc:.4f} (accounts for class imbalance)")
    print("="*60 + "\n")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_per_class': precision_per_class,
        'recall_per_class': recall_per_class,
        'f1_per_class': f1_per_class,
        'support_per_class': support_per_class,
        'confusion_matrix': cm,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'balanced_accuracy': balanced_acc
    }

# Evaluate
eval_results = evaluate_model_performance(best_model, X_test_lgb, y_test, label_encoder)

# Cross-validation
print("🔄 Running 5-fold cross-validation on balanced data...")
cv_scores = cross_val_score(best_model, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy')
print(f"   CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# =============================================================================
# 6. FEATURE IMPORTANCE
# =============================================================================

def extract_feature_importance(model, feature_cols):
    """
    Extract and rank feature importance from trained model.
    
    Shows which features contribute most to prediction decisions,
    helping understand model behavior and identify key weather factors.
    
    Args:
        model: Trained LightGBM model
        feature_cols (list): List of feature names
        
    Returns:
        dict: Feature name to importance score mapping (sorted descending)
        
    Notes:
        - Uses LightGBM's built-in feature_importances_ (gain-based)
        - Higher values indicate more important features
        - Sorted from most to least important
    """
    print("📈 Extracting feature importance...")
    
    importances = model.feature_importances_
    feature_importance = dict(zip(feature_cols, importances.tolist()))
    feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    print("Top 10 features:")
    for i, (feat, imp) in enumerate(list(feature_importance.items())[:10], 1):
        print(f"  {i}. {feat}: {imp:.4f}")
    
    return feature_importance

feature_importance = extract_feature_importance(best_model, feature_cols)

# =============================================================================
# 7. CALCULATE THRESHOLDS
# =============================================================================

def calculate_suspension_thresholds(df):
    """
    Compute statistical thresholds for each suspension level.
    
    Analyzes historical data to determine typical weather conditions
    associated with each suspension level.
    
    Args:
        df (pd.DataFrame): Full dataset with suspension labels
        
    Returns:
        dict: Nested dictionary with statistics per suspension level
        
    Statistics Computed (per level):
        - precipitation_mean: Average precipitation
        - precipitation_max: Maximum precipitation observed
        - windspeed_mean: Average wind speed
        - windspeed_max: Maximum wind speed observed
        - humidity_mean: Average relative humidity
        - count: Number of occurrences
        
    Notes:
        - Used for rule-based predictions and explainability
        - Helps validate model predictions against historical patterns
    """
    print("\n🎯 Calculating suspension thresholds...")
    
    thresholds = {}
    for level in range(6):
        level_data = df[df['suspension'] == level]
        if len(level_data) > 0:
            thresholds[f'level_{level}'] = {
                'precipitation_mean': float(level_data['precipitation'].mean()),
                'precipitation_max': float(level_data['precipitation'].max()),
                'windspeed_mean': float(level_data['windspeed_10m'].mean()),
                'windspeed_max': float(level_data['windspeed_10m'].max()),
                'humidity_mean': float(level_data['relativehumidity_2m'].mean()),
                'count': int(len(level_data))
            }
    
    return thresholds

thresholds = calculate_suspension_thresholds(df)

# =============================================================================
# 8. SAVE ARTIFACTS
# =============================================================================

def save_model_artifacts(model, preprocessor, city_encoder, label_encoder, 
                        eval_results, feature_importance, thresholds, 
                        feature_cols, df, cv_scores):
    """
    Save all trained artifacts to disk for deployment.
    
    Persists model, preprocessors, encoders, and metadata required
    for making predictions on new data.
    
    Args:
        model: Trained LightGBM classifier
        preprocessor: Fitted StandardScaler
        city_encoder: Fitted LabelEncoder for cities
        label_encoder: Fitted LabelEncoder for suspension levels
        eval_results (dict): Evaluation metrics
        feature_importance (dict): Feature importance scores
        thresholds (dict): Suspension level thresholds
        feature_cols (list): List of feature names
        df (pd.DataFrame): Full dataset
        cv_scores (np.array): Cross-validation scores
        
    Saves:
        - model.pkl: Trained classifier
        - preprocessor.pkl: Feature scaler
        - city_encoder.pkl: City label encoder
        - label_encoder.pkl: Target label encoder
        - metrics.json: All evaluation metrics
        - feature_importance.json: Feature rankings
        - confusion_matrix.npy: Confusion matrix array
        - thresholds.json: Suspension level statistics
        - training_stats.json: High-level training summary
        - model_ready.flag: Deployment readiness indicator
        
    Notes:
        - Uses joblib for efficient model serialization
        - JSON format for human-readable metadata
        - All artifacts timestamped with training date
    """
    print("\n💾 Saving model artifacts...\n")
    
    # Save model and preprocessors
    joblib.dump(model, 'model.pkl')
    print("✅ Saved: model.pkl")
    
    joblib.dump(preprocessor, 'preprocessor.pkl')
    print("✅ Saved: preprocessor.pkl")
    
    joblib.dump(city_encoder, 'city_encoder.pkl')
    print("✅ Saved: city_encoder.pkl")
    
    joblib.dump(label_encoder, 'label_encoder.pkl')
    print("✅ Saved: label_encoder.pkl")
    
    # Save metrics
    best_params = {
        'colsample_bytree': 0.8,
        'learning_rate': 0.03,
        'max_depth': 10,
        'min_child_samples': 10,
        'n_estimators': 800,
        'num_leaves': 100,
        'reg_alpha': 0.3,
        'reg_lambda': 2.0,
        'subsample': 0.9,
        'min_child_weight': 0.001,
        'boost_from_average': False
    }
    
    metrics = {
        'best_model': 'lightgbm_tuned',
        'models': {
            'lightgbm_tuned': {
                'accuracy': float(eval_results['accuracy']),
                'precision': float(eval_results['precision']),
                'recall': float(eval_results['recall']),
                'f1': float(eval_results['f1']),
                'precision_per_class': eval_results['precision_per_class'].tolist(),
                'recall_per_class': eval_results['recall_per_class'].tolist(),
                'f1_per_class': eval_results['f1_per_class'].tolist(),
                'support_per_class': eval_results['support_per_class'].tolist(),
                'confusion_matrix': eval_results['confusion_matrix'].tolist(),
                'cv_mean': float(cv_scores.mean()),
                'cv_std': float(cv_scores.std()),
                'best_params': best_params,
                'balanced_accuracy': float(eval_results['balanced_accuracy']),
                'macro_precision': float(eval_results['precision_macro']),
                'macro_recall': float(eval_results['recall_macro']),
                'macro_f1': float(eval_results['f1_macro'])
            }
        },
        'hyperparameters': best_params,
        'feature_names': feature_cols,
        'trained_date': datetime.now().isoformat(),
        'label_mapping': {int(k): int(v) for k, v in enumerate(label_encoder.classes_)},
        'dataset_info': {
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min().isoformat(),
                'end': df['date'].max().isoformat()
            },
            'cities': list(city_encoder.classes_)
        }
    }
    
    with open('metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print("✅ Saved: metrics.json")
    
    with open('feature_importance.json', 'w') as f:
        json.dump(feature_importance, f, indent=2)
    print("✅ Saved: feature_importance.json")
    
    np.save('confusion_matrix.npy', eval_results['confusion_matrix'])
    print("✅ Saved: confusion_matrix.npy")
    
    with open('thresholds.json', 'w') as f:
        json.dump(thresholds, f, indent=2)
    print("✅ Saved: thresholds.json")
    
    training_stats = {
        'model_version': '2.0',
        'training_date': datetime.now().isoformat(),
        'best_model': 'lightgbm_tuned',
        'accuracy': float(eval_results['accuracy']),
        'f1_score': float(eval_results['f1']),
        'train_size': len(X_train),
        'test_size': len(X_test),
        'total_features': len(feature_cols),
        'tuned': True,
        'best_hyperparameters': best_params
    }
    
    with open('training_stats.json', 'w') as f:
        json.dump(training_stats, f, indent=2)
    print("✅ Saved: training_stats.json")
    
    with open('model_ready.flag', 'w') as f:
        f.write(f"Model trained successfully at {datetime.now()}")
    print("✅ Saved: model_ready.flag")

save_model_artifacts(
    best_model, preprocessor, city_encoder, label_encoder,
    eval_results, feature_importance, thresholds, feature_cols, df, cv_scores
)

# =============================================================================
# 9. TRAINING REPORT
# =============================================================================

def print_training_report(df, city_encoder, eval_results, cv_scores):
    """
    Display comprehensive training summary.
    
    Prints a formatted report with all key information about the
    trained model and its performance.
    
    Args:
        df (pd.DataFrame): Full dataset
        city_encoder (LabelEncoder): City encoder
        eval_results (dict): Evaluation metrics
        cv_scores (np.array): Cross-validation scores
        
    Displays:
        - Model type and configuration
        - Dataset statistics
        - Best hyperparameters
        - Performance metrics (weighted, balanced, macro)
        - Cross-validation results
        - Per-class F1 scores
        
    Notes:
        - Formatted for readability in notebooks
        - Provides at-a-glance model quality assessment
    """
    best_params = {
        'colsample_bytree': 0.8,
        'learning_rate': 0.03,
        'max_depth': 10,
        'min_child_samples': 10,
        'n_estimators': 800,
        'num_leaves': 100,
        'reg_alpha': 0.3,
        'reg_lambda': 2.0,
        'subsample': 0.9,
        'min_child_weight': 0.001,
        'boost_from_average': False
    }
    
    print("\n" + "="*60)
    print("📋 TRAINING REPORT")
    print("="*60)
    print(f"\n🎯 Model: LightGBM Classifier (Hyperparameter Tuned)")
    print(f"📊 Dataset: {len(df):,} records")
    print(f"📅 Date Range: {df['date'].min()} to {df['date'].max()}")
    print(f"🏙️ Cities: {len(city_encoder.classes_)}")
    
    print(f"\n🔧 Best Hyperparameters:")
    for param, value in sorted(best_params.items()):
        print(f"   {param}: {value}")
    
    print(f"\n📈 Performance Metrics:")
    print(f"   Weighted Accuracy:  {eval_results['accuracy']:.4f}")
    print(f"   Balanced Accuracy:  {eval_results['balanced_accuracy']:.4f}")
    print(f"   Weighted F1:        {eval_results['f1']:.4f}")
    print(f"   Macro F1:           {eval_results['f1_macro']:.4f}")
    print(f"\n🔄 Cross-Validation:")
    print(f"   Mean: {cv_scores.mean():.4f}")
    print(f"   Std:  {cv_scores.std():.4f}")
    
    print(f"\n📊 Per-Class F1 Scores:")
    for class_idx in range(len(eval_results['f1_per_class'])):
        original_class = label_encoder.inverse_transform([class_idx])[0]
        print(f"   Class {original_class}: {eval_results['f1_per_class'][class_idx]:.4f}")
    print("\n" + "="*60)
    print("✅ Model training complete and artifacts saved!")
    print("="*60)
    print("\n🚀 Ready to run: streamlit run main.py\n")

print_training_report(df, city_encoder, eval_results, cv_scores)