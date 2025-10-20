"""
Metro Manila Suspension Prediction - Model Training (Notebook Version)
Simplified workflow for Jupyter notebooks
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

print("📊 Loading dataset...")
df = pd.read_csv('metro_manila_weather_sus_data.csv', parse_dates=['date'])
print(f"✅ Loaded {len(df):,} records from {df['date'].min()} to {df['date'].max()}")

# Handle missing values
print("🔧 Handling missing values...")
df = df.fillna({
    'relativehumidity_2m': df['relativehumidity_2m'].median(),
    'temperature_2m': df['temperature_2m'].median(),
    'precipitation': 0,
    'apparent_temperature': df['apparent_temperature'].median(),
    'windspeed_10m': df['windspeed_10m'].median()
})

print(f"✅ Dataset shape: {df.shape}")
print(f"📈 Suspension distribution:\n{df['suspension'].value_counts().sort_index()}")

# =============================================================================
# 2. FEATURE ENGINEERING
# =============================================================================

print("\n🛠️ Engineering features...")

# Temporal features
df['hour'] = df['date'].dt.hour
df['day_of_week'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_rush_hour'] = df['hour'].isin([7, 8, 9, 17, 18, 19]).astype(int)

# Sort by city and date for rolling features
df = df.sort_values(['city', 'date'])

# Rolling averages by city
print("  📊 Computing rolling averages...")
for window in [3, 6, 12]:
    df[f'precip_roll_{window}h'] = df.groupby('city')['precipitation'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    df[f'wind_roll_{window}h'] = df.groupby('city')['windspeed_10m'].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )

# Lag features
print("  🔄 Creating lag features...")
for lag in [1, 2, 3]:
    df[f'precip_lag_{lag}h'] = df.groupby('city')['precipitation'].shift(lag).fillna(0)
    df[f'temp_lag_{lag}h'] = df.groupby('city')['temperature_2m'].shift(lag).fillna(df['temperature_2m'].median())

# Peak indicators
df['is_precip_peak'] = (df['precipitation'] > df['precip_roll_6h'] * 1.5).astype(int)
df['is_wind_peak'] = (df['windspeed_10m'] > df['wind_roll_6h'] * 1.3).astype(int)

# Temperature delta
df['apparent_temp_delta'] = df['apparent_temperature'] - df['temperature_2m']

# Categorical features
df['precip_intensity'] = pd.cut(df['precipitation'], 
                                bins=[-1, 0, 5, 15, 50, 200],
                                labels=[0, 1, 2, 3, 4]).astype(int)

df['wind_category'] = pd.cut(df['windspeed_10m'],
                             bins=[-1, 20, 40, 60, 100],
                             labels=[0, 1, 2, 3]).astype(int)

# Encode city
print("  🏙️ Encoding cities...")
city_encoder = LabelEncoder()
df['city_encoded'] = city_encoder.fit_transform(df['city'])

print(f"✅ Feature engineering complete. Total features: {len(df.columns)}")

# =============================================================================
# 3. PREPARE TRAIN-TEST SPLIT
# =============================================================================

print("\n✂️ Splitting data (80-20 stratified)...")

# Sort by date first (for reference)
df = df.sort_values('date')

# Select features
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

# Stratified split - ensures all classes in both train and test
print("  🔀 Performing stratified split (preserves class distribution)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42, shuffle=True
)

# Scale numerical features
print("  📏 Scaling features...")
preprocessor = StandardScaler()
numerical_cols = ['relativehumidity_2m', 'temperature_2m', 'precipitation', 
                 'apparent_temperature', 'windspeed_10m', 'apparent_temp_delta']

X_train[numerical_cols] = preprocessor.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = preprocessor.transform(X_test[numerical_cols])

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

print("\n🤖 Training model with class imbalance handling...\n")

# Convert to float for LightGBM
X_train_lgb = X_train.astype(float)
X_test_lgb = X_test.astype(float)

# Analyze class distribution
print("📊 Class Distribution Analysis:")
class_counts = pd.Series(y_train).value_counts().sort_index()
total_samples = len(y_train)
for class_idx, count in class_counts.items():
    percentage = (count / total_samples) * 100
    original_class = label_encoder.inverse_transform([class_idx])[0]
    print(f"   Class {original_class} (encoded {class_idx}): {count:,} samples ({percentage:.2f}%)")

# Calculate class weights (inverse frequency with MORE aggressive weighting)
print("\n⚖️ Calculating class weights...")
class_weights = {}
for class_idx in np.unique(y_train):
    # Use linear scale instead of sqrt for stronger minority class emphasis
    class_weights[class_idx] = total_samples / (len(np.unique(y_train)) * class_counts[class_idx])

print("Class weights (higher = more weight for minority classes):")
for class_idx, weight in sorted(class_weights.items()):
    original_class = label_encoder.inverse_transform([class_idx])[0]
    print(f"   Class {original_class}: {weight:.2f}")

# Manual oversampling of minority classes (alternative to SMOTE)
print("\n🔄 Oversampling minority classes (aggressive balancing)...")

# Convert to numpy arrays for easier manipulation
X_train_array = X_train_lgb.values
y_train_array = y_train

# Separate by class
X_by_class = {i: X_train_array[y_train_array == i] for i in np.unique(y_train_array)}
y_by_class = {i: y_train_array[y_train_array == i] for i in np.unique(y_train_array)}

# Find target size - make it MORE balanced (80% of majority class)
max_samples = max([len(X_by_class[i]) for i in X_by_class.keys()])
target_samples = int(max_samples * 0.8)  # Increased from 50% to 80%

print(f"   Target samples per class: {target_samples:,}")

# Oversample each minority class with random duplication + noise
X_train_balanced = []
y_train_balanced = []

for class_idx in sorted(X_by_class.keys()):
    X_class = X_by_class[class_idx]
    y_class = y_by_class[class_idx]
    
    if len(X_class) < target_samples:
        # Calculate how many duplicates we need
        n_duplicates = target_samples - len(X_class)
        
        # Random sampling with replacement
        duplicate_indices = np.random.choice(len(X_class), size=n_duplicates, replace=True)
        X_duplicates = X_class[duplicate_indices].copy()
        
        # Add small noise to duplicates to create variation
        # Increase noise slightly for more variation
        noise = np.random.normal(0, 0.02, X_duplicates.shape)
        X_duplicates = X_duplicates + noise
        
        # Combine original and duplicates
        X_combined = np.vstack([X_class, X_duplicates])
        y_combined = np.concatenate([y_class, np.full(n_duplicates, class_idx)])
    else:
        X_combined = X_class
        y_combined = y_class
    
    X_train_balanced.append(X_combined)
    y_train_balanced.append(y_combined)

# Stack all classes
X_train_balanced = np.vstack(X_train_balanced)
y_train_balanced = np.concatenate(y_train_balanced)

# Shuffle the balanced dataset
shuffle_idx = np.random.permutation(len(X_train_balanced))
X_train_balanced = X_train_balanced[shuffle_idx]
y_train_balanced = y_train_balanced[shuffle_idx]

print(f"   Original training size: {len(X_train_lgb):,}")
print(f"   Balanced training size: {len(X_train_balanced):,}")
print("\nBalanced class distribution:")
balanced_counts = pd.Series(y_train_balanced).value_counts().sort_index()
for class_idx, count in balanced_counts.items():
    original_class = label_encoder.inverse_transform([class_idx])[0]
    print(f"   Class {original_class}: {count:,} samples")

# Define best hyperparameters with MORE aggressive settings for imbalanced data
best_params = {
    'colsample_bytree': 0.8,
    'learning_rate': 0.03,  # Reduced for more careful learning
    'max_depth': 10,  # Increased to capture complex patterns
    'min_child_samples': 10,  # Reduced to allow smaller leaf nodes
    'n_estimators': 800,  # Increased for more iterations
    'num_leaves': 100,  # Increased for more complex trees
    'reg_alpha': 0.3,  # Reduced regularization
    'reg_lambda': 2.0,  # Reduced regularization
    'subsample': 0.9,
    'min_child_weight': 0.001,  # Allow more sensitive splits
    'boost_from_average': False  # Don't bias toward average (majority class)
}

print("\n🔧 Training with optimized parameters...")
print("="*60)
for param, value in sorted(best_params.items()):
    print(f"   {param}: {value}")
print("="*60)

# Train model with balanced data and class weights
best_model = LGBMClassifier(
    **best_params,
    num_class=len(np.unique(y_train)),
    class_weight=class_weights,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
    force_row_wise=True
)

print("\n🎯 Training on balanced dataset...")
best_model.fit(X_train_balanced, y_train_balanced)
print("✅ Training complete!\n")

# =============================================================================
# 5. EVALUATE MODEL (WITH CLASS-SPECIFIC METRICS)
# =============================================================================

print("📊 Evaluating model on imbalanced test set...\n")

y_pred = best_model.predict(X_test_lgb)
y_pred_proba = best_model.predict_proba(X_test_lgb)

# Overall metrics
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(
    y_test, y_pred, average='weighted', zero_division=0
)

# Per-class metrics (IMPORTANT for imbalanced data)
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

# Confusion matrix with detailed breakdown
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

# Macro vs Weighted metrics (important distinction for imbalanced data)
print("\n" + "="*60)
print("📊 AGGREGATE METRICS")
print("="*60)
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    y_test, y_pred, average='macro', zero_division=0
)

print(f"\nWeighted Metrics (accounts for class frequency):")
print(f"   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1 Score:  {f1:.4f}")

print(f"\nMacro Metrics (treats all classes equally):")
print(f"   Precision: {precision_macro:.4f}")
print(f"   Recall:    {recall_macro:.4f}")
print(f"   F1 Score:  {f1_macro:.4f}")

# Cross-validation on original training data
print("\n🔄 Running 5-fold cross-validation on balanced data...")
cv_scores = cross_val_score(best_model, X_train_balanced, y_train_balanced, cv=5, scoring='accuracy')
print(f"   CV Score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Calculate balanced accuracy (better metric for imbalanced data)
from sklearn.metrics import balanced_accuracy_score
balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f"\n⚖️ Balanced Accuracy: {balanced_acc:.4f} (accounts for class imbalance)")
print("="*60 + "\n")

# =============================================================================
# 6. FEATURE IMPORTANCE
# =============================================================================

print("📈 Extracting feature importance...")

importances = best_model.feature_importances_
feature_importance = dict(zip(feature_cols, importances.tolist()))
feature_importance = dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))

print("Top 10 features:")
for i, (feat, imp) in enumerate(list(feature_importance.items())[:10], 1):
    print(f"  {i}. {feat}: {imp:.4f}")

# =============================================================================
# 7. CALCULATE THRESHOLDS
# =============================================================================

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

# =============================================================================
# 8. SAVE ARTIFACTS
# =============================================================================

print("\n💾 Saving model artifacts...\n")

# Save model and preprocessors
joblib.dump(best_model, 'model.pkl')
print("✅ Saved: model.pkl")

joblib.dump(preprocessor, 'preprocessor.pkl')
print("✅ Saved: preprocessor.pkl")

joblib.dump(city_encoder, 'city_encoder.pkl')
print("✅ Saved: city_encoder.pkl")

joblib.dump(label_encoder, 'label_encoder.pkl')
print("✅ Saved: label_encoder.pkl")

# Save metrics
metrics = {
    'best_model': 'lightgbm_tuned',
    'models': {
        'lightgbm_tuned': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'precision_per_class': precision_per_class.tolist(),
            'recall_per_class': recall_per_class.tolist(),
            'f1_per_class': f1_per_class.tolist(),
            'support_per_class': support_per_class.tolist(),
            'confusion_matrix': cm.tolist(),
            'cv_mean': float(cv_scores.mean()),
            'cv_std': float(cv_scores.std()),
            'best_params': best_params,
            'balanced_accuracy': float(balanced_acc),
            'macro_precision': float(precision_macro),
            'macro_recall': float(recall_macro),
            'macro_f1': float(f1_macro)
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

np.save('confusion_matrix.npy', cm)
print("✅ Saved: confusion_matrix.npy")

with open('thresholds.json', 'w') as f:
    json.dump(thresholds, f, indent=2)
print("✅ Saved: thresholds.json")

training_stats = {
    'model_version': '2.0',
    'training_date': datetime.now().isoformat(),
    'best_model': 'lightgbm_tuned',
    'accuracy': float(accuracy),
    'f1_score': float(f1),
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

# =============================================================================
# 9. TRAINING REPORT
# =============================================================================

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
print(f"   Weighted Accuracy:  {accuracy:.4f}")
print(f"   Balanced Accuracy:  {balanced_acc:.4f}")
print(f"   Weighted F1:        {f1:.4f}")
print(f"   Macro F1:           {f1_macro:.4f}")
print(f"\n🔄 Cross-Validation:")
print(f"   Mean: {cv_scores.mean():.4f}")
print(f"   Std:  {cv_scores.std():.4f}")

print(f"\n📊 Per-Class F1 Scores:")
for class_idx in range(len(f1_per_class)):
    original_class = label_encoder.inverse_transform([class_idx])[0]
    print(f"   Class {original_class}: {f1_per_class[class_idx]:.4f}")
print("\n" + "="*60)
print("✅ Model training complete and artifacts saved!")
print("="*60)
print("\n🚀 Ready to run: streamlit run main.py\n")