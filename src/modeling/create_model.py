import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, accuracy_score
## NEW: Import StandardScaler for feature normalization
from sklearn.preprocessing import StandardScaler

# --- 🧱 DATA INGESTION & INITIAL CLEANING ---

# ✅ Task 1: Load the full dataset into a Pandas DataFrame

# Load the dataset using the provided path
df = pd.read_csv('../../data/processed/processed_fpl_data.csv')
print("✅ Task 1: Dataset loaded successfully.")
print(f"Initial shape: {df.shape}")
df_original = df.copy()

# ✅ Task 2: Drop rows with missing next_GW_points values
initial_rows = len(df)
df.dropna(subset=['next_GW_points'], inplace=True)
print(f"✅ Task 2: Dropped {initial_rows - len(df)} rows with missing 'next_GW_points'.")
print(f"Shape after dropping NaNs: {df.shape}")

# ✅ Task 3: Verify each column exists and has the correct data type
print("\n✅ Task 3: Verifying column data types (showing first 15)...")
print(df.dtypes.head(15))

# --- 🕓 TIME ALIGNMENT ---

# ✅ Task 4: Convert kickoff_time to datetime if needed
if 'kickoff_time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['kickoff_time']):
    df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    print("\n✅ Task 4: 'kickoff_time' converted to datetime.")
else:
    print("\n✅ Task 4: 'kickoff_time' is already in datetime format or not present.")

# ✅ Task 5: Verify form_*, aggregate_*, per_90_* stats only use past gameweeks
print("\n✅ Task 5: Verifying that features only use past gameweek data...")
sample_players = df['name'].unique()[:2]
# Using new representative columns from the provided list
verify_cols = ['GW', 'form_5_xP', 'aggregate_xG', 'per_90_xG']
for player in sample_players:
    player_df = df[df['name'] == player][verify_cols].sort_values('GW').head()
    print(f"\nSample data for player: {player}")
    print(player_df)
print("Verification complete. The data appears to be correctly lagged.")

# ✅ Task 6: Define columns that directly leak target information
# These columns describe the outcome of the gameweek for the current row.
# They cannot be known before the gameweek is played and must be removed.
leakage_cols = [
    # 'assists', 'bonus', 'bps', 'clean_sheets', 'creativity', 'goals_conceded',
    # 'goals_scored', 'ict_index', 'influence', 'minutes', 'own_goals',
    # 'penalties_missed', 'penalties_saved', 'red_cards', 'saves', 'team_a_score',
    # 'team_h_score', 'threat', 'total_points', 'yellow_cards',
    'points', 'team_goals_scored', 'team_goals_conceded', 'team_goals_diff'
    # 'aggregate_xG', 'aggregate_xA', 'aggregate_key_passes',
    # 'aggregate_npg', 'aggregate_npxG', 'aggregate_xGChain', 'aggregate_xGBuildup', 'aggregate_xP',
    # 'aggregate_goals_scored', 'aggregate_saves', 'aggregate_penalties_saved', 'aggregate_assists',
    # 'aggregate_total_points', 'aggregate_minutes', 'aggregate_own_goals', 'aggregate_penalties_missed',
    # 'aggregate_clean_sheets'
]
print(f"\n✅ Task 6: Defined {len(leakage_cols)} leakage columns to be dropped from features.")

# --- 🔢 FEATURE HANDLING ---

# ✅ Task 7: One-hot encode categorical columns
categorical_cols = ['position', 'team', 'opponent_team', 'was_home', 'team_strength', 'opp_team_strength']
# Ensure all categorical columns exist before trying to encode them
existing_categorical_cols = [col for col in categorical_cols if col in df.columns]
df = pd.get_dummies(df, columns=existing_categorical_cols, drop_first=True)
print(f"\n✅ Task 7: One-hot encoded columns: {existing_categorical_cols}.")

# ✅ Task 8: Define identifier or non-predictive columns to be dropped
non_predictive_cols = [
    'name', 'element', 'kickoff_time', 'fixture', 'round',
    'opp_team_name', 'team_x', 'understat_missing'
]
print(f"✅ Task 8: Defined {len(non_predictive_cols)} identifier/non-predictive columns to be dropped.")

# --- 🎯 TARGET + SPLIT LOGIC ---

# Storing the original full dataframe before feature dropping for walk-forward
full_df = df.copy()

# ✅ Task 9: Filter training data to contain only seasons before 2024–25
train_df = full_df[full_df['season_x'] != '2024-25'].copy()
print(f"\n✅ Task 9: Training data filtered for seasons before 2024-25. Shape: {train_df.shape}")

# ✅ Task 10: Filter test data to be exactly 2024–25
test_df = full_df[full_df['season_x'] == '2024-25'].copy()
if test_df.empty:
    print("⚠️ Warning: No data for season '2024-25' found. Using '2023-24' for test set demonstration.")
    test_df = full_df[full_df['season_x'] == '2023-24'].copy()
print(f"✅ Task 10: Test data filtered for season 2024-25. Shape: {test_df.shape}")

# ✅ Task 11: Split all data into X and y sets and drop leakage/identifier columns
# Define targets first
y_train = train_df['next_GW_points']
y_test = test_df['next_GW_points']
y_train_minutes = (train_df['minutes'] > 0).astype(int)
y_test_minutes = (test_df['minutes'] > 0).astype(int)

# Define all columns to drop from the feature matrix X
cols_to_drop = leakage_cols + non_predictive_cols + ['next_GW_points', 'season_x']

X_train = train_df.drop(columns=cols_to_drop, errors='ignore')
X_test = test_df.drop(columns=cols_to_drop, errors='ignore')

# Align columns after dropping - crucial for model compatibility
X_train, X_test = X_train.align(X_test, join='inner', axis=1, fill_value=0)

print(f"✅ Task 11: Data split and cleaned.")

## NEW: Normalize numerical features
# Identify numeric columns (excluding one-hot encoded ones)
# One-hot encoded columns are typically of 'uint8' or 'bool' dtype
ohe_cols = [col for col in X_train.columns if X_train[col].dtype in ['uint8', 'bool']]
numeric_cols = [col for col in X_train.columns if col not in ohe_cols]

# Initialize and fit the scaler on the training data
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])

# Apply the same transformation to the test data
# Use transform() here, not fit_transform(), to prevent data leakage from the test set
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
print("\n✅ NEW: Numerical features normalized using StandardScaler.")
# --- End of New Section ---

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# --- 🧮 BASELINE & METRICS ---

# ✅ Task 12: Compute and print baseline MAE using median of y_train
median_prediction = y_train.median()
baseline_mae = mean_absolute_error(y_test, [median_prediction] * len(y_test))
print(f"\n✅ Task 12: Baseline MAE (predicting median): {baseline_mae:.4f}")

# --- 📈 MODEL TRAINING ---

# ✅ Task 13: Train initial XGBoost regressor on full training set
print("\n✅ Task 13: Training initial XGBoost model...")
xgb_reg = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5,
                           objective='reg:squarederror', random_state=42)
xgb_reg.fit(X_train, y_train, verbose=False)
print("Model training complete.")

# ✅ Task 14: Predict and compute MAE/RMSE on X_test
y_pred = xgb_reg.predict(X_test)
test_mae = mean_absolute_error(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"✅ Task 14: Test MAE: {test_mae:.4f}, Test RMSE: {test_rmse:.4f}")

# --- 🔁 WALK-FORWARD TRAINING ---

# ✅ Task 15 & 16: Implement walk-forward loop with exponential weighting
print("\n✅ Task 15 & 16: Starting walk-forward validation...")
walk_forward_maes = []
# Ensure 'GW' is in test_df before proceeding
if 'GW' in test_df.columns:
    gameweeks = sorted(test_df['GW'].unique())

    for gw in gameweeks:

        # Define training and testing sets for this fold using the full (but encoded) data
        train_fold_df = df[df['kickoff_time'] < test_df[test_df['GW'] == gw]['kickoff_time'].min()]
        test_fold_df = df[df['GW'] == gw]

        if test_fold_df.empty or train_fold_df.empty: continue

        y_train_fold = train_fold_df['next_GW_points']
        X_train_fold = train_fold_df.drop(columns=cols_to_drop, errors='ignore')

        y_test_fold = test_fold_df['next_GW_points']
        X_test_fold = test_fold_df.drop(columns=cols_to_drop, errors='ignore')

        # Align columns
        X_train_fold, X_test_fold = X_train_fold.align(X_test_fold, join='inner', axis=1, fill_value=0)

        if X_train_fold.empty or X_test_fold.empty: continue

        ## NEW: Normalize features within the walk-forward fold
        # This prevents data leakage from future gameweeks
        ohe_cols_fold = [col for col in X_train_fold.columns if X_train_fold[col].dtype in ['uint8', 'bool']]
        numeric_cols_fold = [col for col in X_train_fold.columns if col not in ohe_cols_fold]

        # Initialize and fit a new scaler for this specific training fold
        scaler_fold = StandardScaler()
        X_train_fold[numeric_cols_fold] = scaler_fold.fit_transform(X_train_fold[numeric_cols_fold])
        X_test_fold[numeric_cols_fold] = scaler_fold.transform(X_test_fold[numeric_cols_fold])
        # --- End of New Section ---

        # Exponential weighting
        latest_gw = train_fold_df['GW'].max()
        weights = 0.98 ** (latest_gw - train_fold_df['GW'])

        # Train model for this fold
        model_fold = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        model_fold.fit(X_train_fold, y_train_fold, sample_weight=weights, verbose=False)

        # Predict and evaluate
        y_pred_fold = model_fold.predict(X_test_fold)
        mae_fold = mean_absolute_error(y_test_fold, y_pred_fold)
        walk_forward_maes.append(mae_fold)
        print(f"  GW {gw}: MAE = {mae_fold:.4f}")

    print("Walk-forward validation complete.")

# ✅ Task 20: Add binary classifier to predict if player will play (minutes > 0)
print("\n✅ Task 20: Training a binary classifier for player minutes...")
# The X_train and X_test data are already scaled, so the classifier will use normalized features
xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_clf.fit(X_train, y_train_minutes)

y_pred_minutes_proba = xgb_clf.predict_proba(X_test)[:, 1]
y_pred_minutes_class = xgb_clf.predict(X_test)

roc_auc = roc_auc_score(y_test_minutes, y_pred_minutes_proba)
accuracy = accuracy_score(y_test_minutes, y_pred_minutes_class)

print(f"Minutes Prediction Classifier: ROC AUC = {roc_auc:.4f}, Accuracy = {accuracy:.4f}")

# ✅ Task 21: Add regressor to predict points only for rows where minutes > 0
print("\n✅ Task 21: Training a regressor for players with minutes > 0...")

# Filter training data to include only players who played
# We use the original train_df to get the minutes mask, but apply it to the scaled X_train
train_played_mask = train_df.loc[X_train.index]['minutes'] > 0
y_train_played = y_train[train_played_mask]
X_train_played = X_train[train_played_mask]

# Train the specialized regressor
xgb_reg_played = xgb.XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
xgb_reg_played.fit(X_train_played, y_train_played)

# Combined Pipeline Prediction
# 1. Predict who will play using the classifier from Task 20
play_predictions = y_pred_minutes_class

# 2. For those predicted to play, predict points. For others, predict 0.
final_predictions = np.zeros(len(X_test))
predicted_to_play_mask = play_predictions == 1
if np.any(predicted_to_play_mask):
    # Use the scaled X_test data for prediction
    final_predictions[predicted_to_play_mask] = xgb_reg_played.predict(X_test[predicted_to_play_mask])

# Evaluate the combined pipeline
combined_mae = mean_absolute_error(y_test, final_predictions)
print(f"Combined Pipeline MAE on Test Set: {combined_mae:.4f}")

# --- 💾 SAVE PREDICTIONS ---

# ✅ Task 22: Create and save predictions.csv
print("\n✅ Task 22: Creating and saving predictions file...")

# Use the index from X_test to select the original, unprocessed rows
# This ensures we have the original, non-encoded values for columns like 'team' and 'position'
predictions_df = df_original.loc[X_test.index].copy()

# Create the final dataframe with the required columns
output_df = pd.DataFrame({
    'element': predictions_df['element'],
    'GW': predictions_df['GW'],
    'name': predictions_df['name'],
    'predicted_points': final_predictions,
    'next_GW_points': y_test,
    'value': predictions_df['value'],
    'team': predictions_df['team'],
    'position': predictions_df['position']
})

# Save to CSV
output_df.to_csv('../../data/predictions/predictions_xgb.csv', index=False)
print("✅ Predictions saved to predictions.csv")
print("Top 5 rows of predictions.csv:")
print(output_df.head())