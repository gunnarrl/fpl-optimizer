import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import os

# --- ⚙️ CONFIGURATION ---
PLANNING_HORIZON = 6
MODEL_OUTPUT_DIR = '../../data/output/models/xgb'
N_ITER_SEARCH = 10
CV_FOLDS = 5
WEIGHT_DECAY = 0.98  # exponential decay factor
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

os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
print(f"✅ Models will be saved to: {MODEL_OUTPUT_DIR}")

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
                  'opp_team_name', 'team_x', 'understat_missing']

# --- 🎯 Walk-forward per GW FOLD TRAINING for 2024-25 ---
season = '2024-25'

if 'GW' in df.columns:
    target_gws = sorted(df[df['season_x'] == season]['GW'].unique())
    for gw in target_gws:
        print(f"\n--- Fold for GW {gw} ({season}) ---")
        train_fold = df[(df['season_x'] != season) | (df['GW'] < gw)]
        test_fold = df[(df['season_x'] == season) & (df['GW'] == gw)]
        if train_fold.empty or test_fold.empty:
            print(f"Skipping GW {gw}: insufficient data.")
            continue

        # For each horizon
        for i in range(1, PLANNING_HORIZON + 1):
            pts_col = f'points_gw+{i}'
            min_col = f'minutes_gw+{i}'
            if pts_col not in train_fold.columns or pts_col not in test_fold.columns:
                continue

            train_i = train_fold.dropna(subset=[pts_col, min_col])
            test_i = test_fold.dropna(subset=[pts_col, min_col])
            if train_i.empty or test_i.empty:
                continue

            # --- Prepare feature sets without leakage ---
            # Classifier: drop all future points and minutes columns
            all_future = [f'points_gw+{j}' for j in range(1, PLANNING_HORIZON+1)] + \
                         [f'minutes_gw+{j}' for j in range(1, PLANNING_HORIZON+1)]
            base_drop = all_future + ['season_x', 'absolute_GW'] + non_predictive
            X_train_clf = train_i.drop(columns=base_drop, errors='ignore')
            X_test_clf = test_i.drop(columns=base_drop, errors='ignore')

            # Regressor: keep only minutes_gw+1, drop other future columns
            keep = ['minutes_gw+1']
            reg_drop = all_future + ['season_x', 'absolute_GW'] + non_predictive
            X_train_reg = train_i.drop(columns=[c for c in reg_drop if c not in keep], errors='ignore')
            X_test_reg = test_i.drop(columns=[c for c in reg_drop if c not in keep], errors='ignore')

            # Targets
            y_train_pts = train_i[pts_col]
            y_train_min = (train_i[min_col] >= 60).astype(int)
            y_test_pts = test_i[pts_col]
            y_test_min = (test_i[min_col] >= 60).astype(int)

            # Align and scale classifier set
            X_train_clf, X_test_clf = X_train_clf.align(X_test_clf, join='inner', axis=1, fill_value=0)
            ohe = [c for c in X_train_clf.columns if X_train_clf[c].dtype in ['uint8', 'bool']]
            num = [c for c in X_train_clf.columns if c not in ohe]
            scaler_clf = StandardScaler()
            X_train_clf[num] = scaler_clf.fit_transform(X_train_clf[num])
            X_test_clf[num] = scaler_clf.transform(X_test_clf[num])

            # Align and scale regressor set
            X_train_reg, X_test_reg = X_train_reg.align(X_test_reg, join='inner', axis=1, fill_value=0)
            ohe_r = [c for c in X_train_reg.columns if X_train_reg[c].dtype in ['uint8', 'bool']]
            num_r = [c for c in X_train_reg.columns if c not in ohe_r]
            scaler_reg = StandardScaler()
            X_train_reg[num_r] = scaler_reg.fit_transform(X_train_reg[num_r])
            X_test_reg[num_r] = scaler_reg.transform(X_test_reg[num_r])

            # Weights
            latest = train_i['absolute_GW'].max()
            weights = WEIGHT_DECAY ** (latest - train_i['absolute_GW'])

            # --- Hyperparameter tuning & train Regressor ---
            print(f"Training regressor for GW{gw}+{i}...")
            xgb_reg = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
            rs_reg = RandomizedSearchCV(
                xgb_reg, param_distributions=PARAM_GRID_REG, n_iter=N_ITER_SEARCH,
                cv=CV_FOLDS, scoring='neg_mean_absolute_error', n_jobs=-1,
                random_state=42, verbose=1
            )
            rs_reg.fit(X_train_reg, y_train_pts, sample_weight=weights)
            best_reg = rs_reg.best_estimator_
            print(f"Best reg params: {rs_reg.best_params_}")
            reg_path = os.path.join(MODEL_OUTPUT_DIR, f'points_predictor_GW{gw}+{i}.json')
            best_reg.save_model(reg_path)
            print(f"✅ Saved regressor: {reg_path}")

            # --- Hyperparameter tuning & train Classifier with calibration ---
            print(f"Training classifier for GW{gw}+{i} with Platt scaling...")
            xgb_clf = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42)
            rs_clf = RandomizedSearchCV(
                xgb_clf, param_distributions=PARAM_GRID_CLF, n_iter=N_ITER_SEARCH,
                cv=CV_FOLDS, scoring='roc_auc', n_jobs=-1,
                random_state=42, verbose=1
            )
            rs_clf.fit(X_train_clf, y_train_min, sample_weight=weights)
            best_clf = rs_clf.best_estimator_
            print(f"Best clf params: {rs_clf.best_params_}")

            calibrated_clf = CalibratedClassifierCV(base_estimator=best_clf, method='sigmoid', cv='prefit')
            calibrated_clf.fit(X_train_clf, y_train_min)

            clf_path = os.path.join(MODEL_OUTPUT_DIR, f'minutes_classifier_GW{gw}+{i}.json')
            best_clf.save_model(clf_path)
            print(f"✅ Saved calibrated classifier: {clf_path}")

    print("\n--- Walk-forward hyperparameter-tuned training complete. ---")
