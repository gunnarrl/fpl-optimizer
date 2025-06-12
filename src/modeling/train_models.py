import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import os

# --- ⚙️ CONFIGURATION ---
PLANNING_HORIZON = 6
MODEL_OUTPUT_DIR = '../../data/output/models'
# Hyperparameter tuning configuration
N_ITER_SEARCH = 10  # Number of parameter settings that are sampled.
CV_FOLDS = 5  # Number of cross-validation folds.

# Create the directory if it doesn't exist
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
print(f"✅ Models will be saved to: {MODEL_OUTPUT_DIR}")

# --- 🧱 DATA INGESTION ---
df = pd.read_csv('../../data/processed/processed_fpl_data.csv')
print("✅ Dataset loaded successfully.")
print(f"Initial shape: {df.shape}")
df_original = df.copy()

# --- 🎯 TARGET VARIABLE CREATION ---
print(f"\n✅ Generating target variables for a planning horizon of {PLANNING_HORIZON} gameweeks...")

future_targets = []
for i in range(1, PLANNING_HORIZON + 1):
    points_col_name = f'points_gw+{i}'
    minutes_col_name = f'minutes_gw+{i}'
    future_targets.extend([points_col_name, minutes_col_name])

df.dropna(subset=future_targets, how='all', inplace=True)
print(f"✅ Successfully identified target labels for a planning horizon of {PLANNING_HORIZON} weeks.")
print(f"Shape after ensuring targets exist: {df.shape}")

# ✅ Verify each column exists and has the correct data type
print("\n✅ Verifying column data types (showing first 15)...")
print(df.dtypes.head(15))

# --- 🕓 TIME ALIGNMENT ---
if 'kickoff_time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['kickoff_time']):
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    print("\n✅ 'kickoff_time' converted to datetime.")
else:
    print("\n✅ 'kickoff_time' is already in datetime format or not present.")

# ✅ Define columns that directly leak target information
leakage_cols = [
    # 'points', 'team_goals_scored', 'team_goals_conceded', 'team_goals_diff'
]
# IMPORTANT: Add all future target columns to the list of columns to be dropped from FEATURES
leakage_cols.extend(future_targets)
print(f"\n✅ Defined {len(leakage_cols)} leakage columns to be dropped from features.")

# --- 🔢 FEATURE HANDLING ---
categorical_cols = ['position', 'team', 'opponent_team', 'opponent_2_team', 'was_home', 'team_strength', 'opp_team_strength',
                    'is_dgw_in_gw+1','is_dgw_in_gw+2','is_dgw_in_gw+3','is_dgw_in_gw+4','is_dgw_in_gw+5',
                    'is_dgw_in_gw+6','is_dgw']
existing_categorical_cols = [col for col in categorical_cols if col in df.columns]
df = pd.get_dummies(df, columns=existing_categorical_cols, drop_first=True)
print(f"\n✅ One-hot encoded columns: {existing_categorical_cols}.")

non_predictive_cols = [
    'name', 'element', 'kickoff_time', 'fixture', 'round',
    'opp_team_name', 'team_x', 'understat_missing', 'opp_team_name_2', 'opp_team_name_1'
]
print(f"✅ Defined {len(non_predictive_cols)} identifier/non-predictive columns to be dropped.")

# --- 🎯 DATA SPLITTING ---
train_df_full = df[df['season_x'] != '2024-25'].copy()
print(f"\n✅ Training data filtered for seasons before 2024-25. Shape: {train_df_full.shape}")

test_df_full = df[df['season_x'] == '2024-25'].copy()
if test_df_full.empty:
    print("⚠️ Warning: No data for season '2024-25' found. Using '2023-24' for test set demonstration.")
    test_df_full = df[df['season_x'] == '2023-24'].copy()
print(f"✅ Test data filtered for season. Shape: {test_df_full.shape}")

# Define all columns to drop from the feature matrix X
# We will now handle feature selection dynamically inside the loop
cols_to_drop_base = list(set(leakage_cols + non_predictive_cols + ['season_x']))

# --- 📈 MODEL TRAINING LOOP ---
print("\n✅ Starting multi-gameweek model training loop with hyperparameter tuning...")

for i in range(1, PLANNING_HORIZON + 1):
    current_gw_str = f"GW+{i}"
    print(f"\n--- Training models for {current_gw_str} ---")

    # Define the specific targets for this iteration
    points_target_col = f'points_gw+{i}'
    minutes_target_col = f'minutes_gw+{i}'

    # Create clean train/test sets for this iteration
    train_df = train_df_full.dropna(subset=[points_target_col, minutes_target_col])
    test_df = test_df_full.dropna(subset=[points_target_col, minutes_target_col])

    # Define targets
    y_train_points = train_df[points_target_col]
    y_test_points = test_df[points_target_col]
    y_train_minutes = (train_df[minutes_target_col] >= 60).astype(int)
    y_test_minutes = (test_df[minutes_target_col] >= 60).astype(int)

    # --- Feature Engineering for this specific horizon ---
    # For all models, drop base non-predictive columns and all future target columns
    X_train_base = train_df.drop(columns=cols_to_drop_base, errors='ignore')
    X_test_base = test_df.drop(columns=cols_to_drop_base, errors='ignore')

    # For points prediction, add back 'minutes_gw+1' as a feature
    # This is a special feature you've allowed based on domain knowledge
    X_train_points = X_train_base.copy()
    X_test_points = X_test_base.copy()
    if 'minutes_gw+1' in train_df.columns:
        X_train_points['minutes_gw+1'] = train_df['minutes_gw+1']
        X_test_points['minutes_gw+1'] = test_df['minutes_gw+1']
        print("✅ Added 'minutes_gw+1' as a feature for the points prediction model.")

    # The minutes predictor features are simply the base features without future info
    X_train_minutes = X_train_base
    X_test_minutes = X_test_base

    # Align columns - crucial for model compatibility after feature adjustments
    X_train_points, X_test_points = X_train_points.align(X_test_points, join='inner', axis=1, fill_value=0)
    X_train_minutes, X_test_minutes = X_train_minutes.align(X_test_minutes, join='inner', axis=1, fill_value=0)


    # --- Preprocessing (Scaling) ---
    def scale_features(X_train, X_test):
        ohe_cols = [col for col in X_train.columns if X_train[col].dtype in ['uint8', 'bool']]
        numeric_cols = [col for col in X_train.columns if col not in ohe_cols]
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
        return X_train_scaled, X_test_scaled


    X_train_points_scaled, X_test_points_scaled = scale_features(X_train_points, X_test_points)
    X_train_minutes_scaled, X_test_minutes_scaled = scale_features(X_train_minutes, X_test_minutes)

    print(f"Points model features prepared: {X_train_points_scaled.shape[1]} columns")
    print(f"Minutes model features prepared: {X_train_minutes_scaled.shape[1]} columns")

    # --- Train Points Regressor with Hyperparameter Tuning ---
    print(f"Tuning and training points prediction model for {current_gw_str}...")
    param_grid_reg = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'colsample_bytree': [0.7, 0.8, 1.0],
        'subsample': [0.7, 0.8, 1.0]
    }
    xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    random_search_reg = RandomizedSearchCV(
        xgb_reg, param_distributions=param_grid_reg, n_iter=N_ITER_SEARCH,
        cv=CV_FOLDS, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=42, verbose=1
    )
    random_search_reg.fit(X_train_points_scaled, y_train_points)
    best_reg = random_search_reg.best_estimator_

    y_pred_points = best_reg.predict(X_test_points_scaled)
    test_mae = mean_absolute_error(y_test_points, y_pred_points)
    print(f"Best Points Model Test MAE for {current_gw_str}: {test_mae:.4f}")
    print(f"Best params: {random_search_reg.best_params_}")

    reg_model_path = os.path.join(MODEL_OUTPUT_DIR, f'model_{current_gw_str}.json')
    best_reg.save_model(reg_model_path)
    print(f"✅ Points prediction model for {current_gw_str} saved to {reg_model_path}")

    # --- Train Minutes Classifier with Hyperparameter Tuning ---
    print(f"Tuning and training minutes prediction classifier for {current_gw_str}...")
    param_grid_clf = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'gamma': [0, 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0]
    }
    xgb_clf = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
    random_search_clf = RandomizedSearchCV(
        xgb_clf, param_distributions=param_grid_clf, n_iter=N_ITER_SEARCH,
        cv=CV_FOLDS, scoring='roc_auc', n_jobs=-1, random_state=42, verbose=1
    )
    random_search_clf.fit(X_train_minutes_scaled, y_train_minutes)
    best_clf = random_search_clf.best_estimator_

    y_pred_minutes_proba = best_clf.predict_proba(X_test_minutes_scaled)[:, 1]
    y_pred_minutes_class = best_clf.predict(X_test_minutes_scaled)
    roc_auc = roc_auc_score(y_test_minutes, y_pred_minutes_proba)
    accuracy = accuracy_score(y_test_minutes, y_pred_minutes_class)
    print(f"Best Minutes Classifier Metrics for {current_gw_str}: ROC AUC = {roc_auc:.4f}, Accuracy = {accuracy:.4f}")
    print(f"Best params: {random_search_clf.best_params_}")

    clf_model_path = os.path.join(MODEL_OUTPUT_DIR, f'minutes_classifier_{current_gw_str}.json')
    best_clf.save_model(clf_model_path)
    print(f"✅ Minutes classifier for {current_gw_str} saved to {clf_model_path}")

print("\n--- All models tuned, trained, and saved successfully. ---")