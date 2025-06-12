import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import pickle
import warnings
from joblib import Parallel, delayed
import time

warnings.filterwarnings('ignore')

# --- ⚙️ CONFIGURATION ---
PLANNING_HORIZON = 6
MODEL_OUTPUT_DIR = '../../data/output/models/xgb'
PREDICTIONS_OUTPUT_DIR = '../../data/output/predictions'
PREDICTIONS_OUTPUT_FILE = 'predictions.csv'
N_ITER_SEARCH = 10
CV_FOLDS = 5
WEIGHT_DECAY = 0.98  # exponential decay factor
N_JOBS_PARALLEL = -1  # Number of parallel jobs (-1 uses all available cores)
PARALLEL_BACKEND = 'threading'  # 'threading' or 'multiprocessing'

# Season-specific gameweek counts
SEASON_LENGTHS = {
    '2019-20': 47,
    # default for other seasons
}
DEFAULT_GW_COUNT = 38

# Hyperparameter grids
PARAM_GRID_REG = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'subsample': [0.7, 0.8, 1.0]
}
PARAM_GRID_CLF = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'gamma': [0, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0]
}

# Create output directories
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_OUTPUT_DIR, exist_ok=True)
print(f"✅ Models will be saved to: {MODEL_OUTPUT_DIR}")
print(f"💾 Predictions will be saved to: {os.path.join(PREDICTIONS_OUTPUT_DIR, PREDICTIONS_OUTPUT_FILE)}")

# --- 🧱 DATA INGESTION & PREP ---
df = pd.read_csv('../../data/processed/processed_fpl_data.csv')
df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])

# Compute cumulative offsets for absolute_GW
seasons = sorted(df['season_x'].unique(), key=lambda x: int(x.split('-')[0]))
offsets = {}
cum = 0
for s in seasons:
    offsets[s] = cum
    length = SEASON_LENGTHS.get(s, DEFAULT_GW_COUNT)
    cum += length

df['absolute_GW'] = df.apply(lambda row: offsets[row['season_x']] + row['GW'], axis=1)

# One-hot encoding
categorical_cols = ['position', 'team', 'opponent_team', 'opponent_2_team', 'was_home',
                    'team_strength', 'opp_team_strength']
for col in categorical_cols:
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], drop_first=True)

non_predictive = ['name', 'element', 'kickoff_time', 'fixture', 'round',
                  'opp_team_name', 'team_x', 'understat_missing', 'opp_team_name_1', 'opp_team_name_2']


def train_and_predict_horizon(i, gw, train_fold, test_fold, non_predictive,
                              PLANNING_HORIZON, MODEL_OUTPUT_DIR, PARAM_GRID_REG,
                              PARAM_GRID_CLF, N_ITER_SEARCH, CV_FOLDS, WEIGHT_DECAY):
    """Train models and make predictions for a single horizon."""

    result = {
        'horizon': i,
        'predicted_points': np.full(len(test_fold), np.nan),
        'actual_points': np.nan,
        'success': False,
        'error': None
    }

    try:
        pts_col = f'points_gw+{i}'
        min_col = f'minutes_gw+{i}'

        if pts_col not in train_fold.columns or pts_col not in test_fold.columns:
            result['error'] = f"Target columns missing for +{i}"
            result['actual_points'] = np.full(len(test_fold), np.nan)
            return result

        # Prepare training data
        train_i = train_fold.dropna(subset=[pts_col, min_col])
        test_i = test_fold.dropna(subset=[pts_col, min_col])

        if train_i.empty:
            result['error'] = f"No training data for +{i}"
            result['actual_points'] = test_fold[pts_col].values if pts_col in test_fold.columns else np.full(
                len(test_fold), np.nan)
            return result

        # --- Prepare feature sets without leakage ---
        all_future = [f'points_gw+{j}' for j in range(1, PLANNING_HORIZON + 1)] + \
                     [f'minutes_gw+{j}' for j in range(1, PLANNING_HORIZON + 1)]

        # Classifier: drop all future columns
        base_drop = all_future + ['season_x', 'absolute_GW'] + non_predictive
        X_train_clf = train_i.drop(columns=base_drop, errors='ignore')
        X_test_clf = test_fold.drop(columns=base_drop, errors='ignore')

        # Regressor: keep only minutes_gw+1, drop other future columns
        keep = ['minutes_gw+1'] if 'minutes_gw+1' in train_i.columns else []
        reg_drop = [c for c in (all_future + ['season_x', 'absolute_GW'] + non_predictive) if c not in keep]
        X_train_reg = train_i.drop(columns=reg_drop, errors='ignore')
        X_test_reg = test_fold.drop(columns=reg_drop, errors='ignore')

        # Targets
        y_train_pts = train_i[pts_col]
        y_train_min = (train_i[min_col] >= 60).astype(int)

        # Align features
        X_train_clf, X_test_clf = X_train_clf.align(X_test_clf, join='inner', axis=1, fill_value=0)
        X_train_reg, X_test_reg = X_train_reg.align(X_test_reg, join='inner', axis=1, fill_value=0)

        # Scale classifier features
        ohe_clf = [c for c in X_train_clf.columns if X_train_clf[c].dtype in ['uint8', 'bool']]
        num_clf = [c for c in X_train_clf.columns if c not in ohe_clf]

        if num_clf:
            scaler_clf = StandardScaler()
            X_train_clf_scaled = X_train_clf.copy()
            X_test_clf_scaled = X_test_clf.copy()
            X_train_clf_scaled[num_clf] = scaler_clf.fit_transform(X_train_clf[num_clf])
            X_test_clf_scaled[num_clf] = scaler_clf.transform(X_test_clf[num_clf])
        else:
            X_train_clf_scaled = X_train_clf
            X_test_clf_scaled = X_test_clf

        # Scale regressor features
        ohe_reg = [c for c in X_train_reg.columns if X_train_reg[c].dtype in ['uint8', 'bool']]
        num_reg = [c for c in X_train_reg.columns if c not in ohe_reg]

        if num_reg:
            scaler_reg = StandardScaler()
            X_train_reg_scaled = X_train_reg.copy()
            X_test_reg_scaled = X_test_reg.copy()
            X_train_reg_scaled[num_reg] = scaler_reg.fit_transform(X_train_reg[num_reg])
            X_test_reg_scaled[num_reg] = scaler_reg.transform(X_test_reg[num_reg])
        else:
            X_train_reg_scaled = X_train_reg
            X_test_reg_scaled = X_test_reg

        # Sample weights
        latest = train_i['absolute_GW'].max()
        weights = WEIGHT_DECAY ** (latest - train_i['absolute_GW'])

        # --- Train Regressor ---
        xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42,
                                   n_jobs=1)  # n_jobs=1 for thread safety
        rs_reg = RandomizedSearchCV(
            xgb_reg, param_distributions=PARAM_GRID_REG, n_iter=N_ITER_SEARCH,
            cv=CV_FOLDS, scoring='neg_mean_absolute_error', n_jobs=1,  # n_jobs=1 for thread safety
            random_state=42, verbose=0
        )
        rs_reg.fit(X_train_reg_scaled, y_train_pts, sample_weight=weights)
        best_reg = rs_reg.best_estimator_

        # Save regressor
        reg_path = os.path.join(MODEL_OUTPUT_DIR, f'points_predictorGW{gw}+{i}.json')
        best_reg.save_model(reg_path)

        # --- Train Classifier ---
        xgb_clf = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=1)
        rs_clf = RandomizedSearchCV(
            xgb_clf, param_distributions=PARAM_GRID_CLF, n_iter=N_ITER_SEARCH,
            cv=CV_FOLDS, scoring='roc_auc', n_jobs=1,  # n_jobs=1 for thread safety
            random_state=42, verbose=0
        )
        rs_clf.fit(X_train_clf_scaled, y_train_min, sample_weight=weights)
        best_clf = rs_clf.best_estimator_

        # Calibrate classifier
        calibrated_clf = CalibratedClassifierCV(base_estimator=best_clf, method='sigmoid', cv=3)
        calibrated_clf.fit(X_train_clf_scaled, y_train_min)

        # Save calibrated classifier
        clf_path = os.path.join(MODEL_OUTPUT_DIR, f'minutes_classifierGW{gw}+{i}.pkl')
        with open(clf_path, 'wb') as f:
            pickle.dump(calibrated_clf, f)

        # --- Make Predictions ---
        test_indices_with_data = test_i.index if not test_i.empty else []

        if len(test_indices_with_data) > 0:
            # Filter scaled test data to match test_i indices
            X_test_clf_pred = X_test_clf_scaled.loc[test_indices_with_data]
            X_test_reg_pred = X_test_reg_scaled.loc[test_indices_with_data]

            # Make predictions
            minutes_proba = calibrated_clf.predict_proba(X_test_clf_pred)[:, 1]
            points_pred = best_reg.predict(X_test_reg_pred)

            # Apply minutes filter
            final_points_pred = np.where(minutes_proba >= 0.5, points_pred, 0)

            # Map back to full test_fold
            prediction_array = np.full(len(test_fold), np.nan)
            test_fold_indices = test_fold.index.get_indexer(test_indices_with_data)
            prediction_array[test_fold_indices] = final_points_pred
        else:
            prediction_array = np.full(len(test_fold), np.nan)

        result['predicted_points'] = prediction_array
        result['actual_points'] = test_fold[pts_col].values
        result['success'] = True

    except Exception as e:
        result['error'] = str(e)
        result['actual_points'] = test_fold[pts_col].values if pts_col in test_fold.columns else np.full(len(test_fold),
                                                                                                         np.nan)

    return result


season = '2024-25'
all_predictions = []

if 'GW' in df.columns:
    target_gws = sorted(df[df['season_x'] == season]['GW'].unique())

    for gw in target_gws:
        print(f"\n--- Training and predicting for GW {gw} ({season}) ---")

        # Split data
        train_fold = df[(df['season_x'] != season) | (df['GW'] < gw)]
        test_fold = df[(df['season_x'] == season) & (df['GW'] == gw)]

        if train_fold.empty or test_fold.empty:
            print(f"Skipping GW {gw}: insufficient data.")
            continue

        # Initialize prediction storage for this GW
        gw_predictions = {
            'element': test_fold['element'].values,
            'GW': test_fold['GW'].values,
            'name': test_fold['name'].values,
            'value': test_fold['value'].values if 'value' in test_fold.columns else np.nan,
            'team': test_fold['team_x'].values if 'team_x' in test_fold.columns else np.nan,
            'position': test_fold['position'].values if 'position' in test_fold.columns else np.nan,
        }

        # For each horizon
        for i in range(1, PLANNING_HORIZON + 1):
            print(f"  🔄 Processing horizon +{i}...")

            pts_col = f'points_gw+{i}'
            min_col = f'minutes_gw+{i}'

            if pts_col not in train_fold.columns or pts_col not in test_fold.columns:
                print(f"    ⚠️ Target columns missing for +{i}, skipping...")
                gw_predictions[f'predicted_points_gw+{i}'] = np.nan
                gw_predictions[f'points_gw+{i}'] = np.nan
                continue

            # Prepare training data
            train_i = train_fold.dropna(subset=[pts_col, min_col])
            test_i = test_fold.dropna(subset=[pts_col, min_col])

            if train_i.empty:
                print(f"    ⚠️ No training data for +{i}, skipping...")
                gw_predictions[f'predicted_points_gw+{i}'] = np.nan
                gw_predictions[f'points_gw+{i}'] = test_fold[pts_col].values if pts_col in test_fold.columns else np.nan
                continue

            # --- Prepare feature sets without leakage ---
            all_future = [f'points_gw+{j}' for j in range(1, PLANNING_HORIZON + 1)] + \
                         [f'minutes_gw+{j}' for j in range(1, PLANNING_HORIZON + 1)]

            # Classifier: drop all future columns
            base_drop = all_future + ['season_x', 'absolute_GW'] + non_predictive
            X_train_clf = train_i.drop(columns=base_drop, errors='ignore')
            X_test_clf = test_fold.drop(columns=base_drop, errors='ignore')

            # Regressor: keep only minutes_gw+1, drop other future columns
            keep = ['minutes_gw+1'] if 'minutes_gw+1' in train_i.columns else []
            reg_drop = [c for c in (all_future + ['season_x', 'absolute_GW'] + non_predictive) if c not in keep]
            X_train_reg = train_i.drop(columns=reg_drop, errors='ignore')
            X_test_reg = test_fold.drop(columns=reg_drop, errors='ignore')

            # Targets
            y_train_pts = train_i[pts_col]
            y_train_min = (train_i[min_col] >= 60).astype(int)

            # Align features
            X_train_clf, X_test_clf = X_train_clf.align(X_test_clf, join='inner', axis=1, fill_value=0)
            X_train_reg, X_test_reg = X_train_reg.align(X_test_reg, join='inner', axis=1, fill_value=0)

            # Scale classifier features
            ohe_clf = [c for c in X_train_clf.columns if X_train_clf[c].dtype in ['uint8', 'bool']]
            num_clf = [c for c in X_train_clf.columns if c not in ohe_clf]

            if num_clf:
                scaler_clf = StandardScaler()
                X_train_clf_scaled = X_train_clf.copy()
                X_test_clf_scaled = X_test_clf.copy()
                X_train_clf_scaled[num_clf] = scaler_clf.fit_transform(X_train_clf[num_clf])
                X_test_clf_scaled[num_clf] = scaler_clf.transform(X_test_clf[num_clf])
            else:
                X_train_clf_scaled = X_train_clf
                X_test_clf_scaled = X_test_clf

            # Scale regressor features
            ohe_reg = [c for c in X_train_reg.columns if X_train_reg[c].dtype in ['uint8', 'bool']]
            num_reg = [c for c in X_train_reg.columns if c not in ohe_reg]

            if num_reg:
                scaler_reg = StandardScaler()
                X_train_reg_scaled = X_train_reg.copy()
                X_test_reg_scaled = X_test_reg.copy()
                X_train_reg_scaled[num_reg] = scaler_reg.fit_transform(X_train_reg[num_reg])
                X_test_reg_scaled[num_reg] = scaler_reg.transform(X_test_reg[num_reg])
            else:
                X_train_reg_scaled = X_train_reg
                X_test_reg_scaled = X_test_reg

            # Sample weights
            latest = train_i['absolute_GW'].max()
            weights = WEIGHT_DECAY ** (latest - train_i['absolute_GW'])

            try:
                # --- Train Regressor ---
                print(f"    📈 Training regressor...")
                xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
                rs_reg = RandomizedSearchCV(
                    xgb_reg, param_distributions=PARAM_GRID_REG, n_iter=N_ITER_SEARCH,
                    cv=CV_FOLDS, scoring='neg_mean_absolute_error', n_jobs=-1,
                    random_state=42, verbose=0
                )
                rs_reg.fit(X_train_reg_scaled, y_train_pts, sample_weight=weights)
                best_reg = rs_reg.best_estimator_

                # Save regressor
                reg_path = os.path.join(MODEL_OUTPUT_DIR, f'points_predictorGW{gw}+{i}.json')
                best_reg.save_model(reg_path)

                # --- Train Classifier ---
                print(f"    🎯 Training classifier...")
                xgb_clf = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
                rs_clf = RandomizedSearchCV(
                    xgb_clf, param_distributions=PARAM_GRID_CLF, n_iter=N_ITER_SEARCH,
                    cv=CV_FOLDS, scoring='roc_auc', n_jobs=-1,
                    random_state=42, verbose=0
                )
                rs_clf.fit(X_train_clf_scaled, y_train_min, sample_weight=weights)
                best_clf = rs_clf.best_estimator_

                # Calibrate classifier
                calibrated_clf = CalibratedClassifierCV(base_estimator=best_clf, method='sigmoid', cv=3)
                calibrated_clf.fit(X_train_clf_scaled, y_train_min)

                # Save calibrated classifier
                clf_path = os.path.join(MODEL_OUTPUT_DIR, f'minutes_classifierGW{gw}+{i}.pkl')
                with open(clf_path, 'wb') as f:
                    pickle.dump(calibrated_clf, f)

                # --- Make Predictions ---
                print(f"    🔮 Making predictions...")

                # Get predictions for test data that has the required features
                test_indices_with_data = test_i.index if not test_i.empty else []

                if len(test_indices_with_data) > 0:
                    # Filter scaled test data to match test_i indices
                    X_test_clf_pred = X_test_clf_scaled.loc[test_indices_with_data]
                    X_test_reg_pred = X_test_reg_scaled.loc[test_indices_with_data]

                    # Make predictions
                    minutes_proba = calibrated_clf.predict_proba(X_test_clf_pred)[:, 1]
                    points_pred = best_reg.predict(X_test_reg_pred)

                    # Apply minutes filter
                    final_points_pred = np.where(minutes_proba >= 0.5, points_pred, 0)

                    # Map back to full test_fold
                    prediction_array = np.full(len(test_fold), np.nan)
                    test_fold_indices = test_fold.index.get_indexer(test_indices_with_data)
                    prediction_array[test_fold_indices] = final_points_pred
                else:
                    prediction_array = np.full(len(test_fold), np.nan)

                gw_predictions[f'predicted_points_gw+{i}'] = prediction_array
                gw_predictions[f'points_gw+{i}'] = test_fold[pts_col].values

                print(f"    ✅ Completed GW{gw}+{i} (predicted {np.sum(~np.isnan(prediction_array))} players)")

            except Exception as e:
                print(f"    ❌ Error in GW{gw}+{i}: {str(e)}")
                gw_predictions[f'predicted_points_gw+{i}'] = np.nan
                gw_predictions[f'points_gw+{i}'] = test_fold[pts_col].values if pts_col in test_fold.columns else np.nan

        # Store predictions for this GW
        all_predictions.append(pd.DataFrame(gw_predictions))

# --- 💾 SAVE PREDICTIONS ---
if all_predictions:
    print(f"\n📊 Combining predictions from {len(all_predictions)} gameweeks...")
    final_predictions_df = pd.concat(all_predictions, ignore_index=True)

    # Reorder columns
    base_cols = ['element', 'GW', 'name', 'value', 'team', 'position']
    pred_cols = [f'predicted_points_gw+{i}' for i in range(1, PLANNING_HORIZON + 1)]
    actual_cols = [f'points_gw+{i}' for i in range(1, PLANNING_HORIZON + 1)]

    column_order = base_cols + pred_cols + actual_cols
    final_predictions_df = final_predictions_df.reindex(columns=column_order)

    # Save predictions
    output_path = os.path.join(PREDICTIONS_OUTPUT_DIR, PREDICTIONS_OUTPUT_FILE)
    final_predictions_df.to_csv(output_path, index=False)

    print(f"✅ Predictions saved to: {output_path}")
    print(f"📈 Shape: {final_predictions_df.shape}")
    print(f"\n📋 Sample predictions:")
    print(final_predictions_df.head())

    # Quick summary stats
    pred_columns = [col for col in final_predictions_df.columns if col.startswith('predicted_points')]
    print(f"\n📊 Prediction Summary:")
    for col in pred_columns:
        non_null = final_predictions_df[col].notna().sum()
        print(f"  {col}: {non_null} predictions made")

else:
    print("❌ No predictions were generated!")

print("\n🎉 Training and prediction complete!")