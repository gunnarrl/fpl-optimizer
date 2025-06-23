import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import pickle
import warnings
import sys
import traceback

## PARALLEL-SPECIFIC CHANGE: Import Dask libraries for parallel execution.
from dask.distributed import Client, LocalCluster
from dask_cuda import LocalCUDACluster
import dask
import logging

# --- 📝 IMPROVED LOGGER CLASS ---
# Use standard logging for better compatibility with HPC schedulers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_log.txt", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

warnings.filterwarnings('ignore')

# --- ⚙️ CONFIGURATION ---
PLANNING_HORIZON = 6
MODEL_OUTPUT_DIR = 'models'
PREDICTIONS_OUTPUT_DIR = '.'
PREDICTIONS_OUTPUT_FILE = 'predictions.csv'
INPUT_DATA_FILE = 'processed_fpl_data.csv'
N_ITER_SEARCH = 10
CV_FOLDS = 5
WEIGHT_DECAY = 0.98
VERBOSE_LEVEL = 0  # Quieter logs for parallel runs
## PARALLEL-SPECIFIC CHANGE: Set N_JOBS for CPU cores per task.
# With 32 CPUs and 4 GPUs (workers), each worker gets 8 cores.
# Setting n_jobs=7 lets each RandomizedSearchCV use most of the cores available to its worker.
N_JOBS_PER_TASK = 7

SEASON_LENGTHS = {'2019-20': 47}
DEFAULT_GW_COUNT = 38

# Random Forest parameter grids
PARAM_GRID_REG = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None]
}
PARAM_GRID_CLF = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}


## PARALLEL-SPECIFIC CHANGE: Encapsulate the logic for a single gameweek into a function.
# Dask will execute this function in parallel for different `gw` values.
def process_gameweek(gw, season, df, config):
    """
    Trains models and generates predictions for a single gameweek.
    This function is designed to be run in parallel by Dask.
    """
    # Unpack config dictionary
    for key, value in config.items():
        globals()[key] = value

    logging.info(f"--- Starting task for GW {gw} ({season}) ---")

    # Split data
    train_fold = df[(df['season_x'] != season) | (df['GW'] < gw)]
    test_fold = df[(df['season_x'] == season) & (df['GW'] == gw)]

    if train_fold.empty or test_fold.empty:
        logging.warning(f"Skipping GW {gw}: insufficient data.")
        return None

    gw_predictions = {
        'element': test_fold['element'].values,
        'GW': test_fold['GW'].values,
        'name': test_fold['name'].values,
        'value': test_fold['value'].values if 'value' in test_fold.columns else np.full(len(test_fold), np.nan),
        'team': test_fold['team_x'].values if 'team_x' in test_fold.columns else np.full(len(test_fold), np.nan),
        'position': test_fold['position'].values if 'position' in test_fold.columns else np.full(len(test_fold),
                                                                                                 np.nan),
    }

    # For each horizon
    for i in range(1, PLANNING_HORIZON + 1):
        logging.info(f"  [GW {gw}] Processing horizon +{i}...")

        pts_col = f'points_gw+{i}'
        min_col = f'minutes_gw+{i}'

        if pts_col not in train_fold.columns or pts_col not in test_fold.columns:
            gw_predictions[f'predicted_points_gw+{i}'] = np.full(len(test_fold), np.nan)
            gw_predictions[f'points_gw+{i}'] = np.full(len(test_fold), np.nan)
            continue

        train_i = train_fold.dropna(subset=[pts_col, min_col])
        test_i = test_fold.dropna(subset=[pts_col, min_col])

        if train_i.empty:
            logging.warning(f"    [GW {gw}] No training data for +{i}, skipping...")
            gw_predictions[f'predicted_points_gw+{i}'] = np.full(len(test_fold), np.nan)
            gw_predictions[f'points_gw+{i}'] = test_fold[pts_col].values
            continue

        # Prepare feature sets without leakage
        all_future = [f'points_gw+{j}' for j in range(1, PLANNING_HORIZON + 1)] + \
                     [f'minutes_gw+{j}' for j in range(1, PLANNING_HORIZON + 1)]
        non_predictive = ['name', 'element', 'kickoff_time', 'fixture', 'round',
                          'opp_team_name', 'team_x', 'understat_missing', 'opp_team_name_1', 'opp_team_name_2']

        base_drop = all_future + ['season_x', 'absolute_GW'] + non_predictive
        X_train_clf = train_i.drop(columns=base_drop, errors='ignore')
        X_test_clf = test_fold.drop(columns=base_drop, errors='ignore')

        keep = ['minutes_gw+1'] if 'minutes_gw+1' in train_i.columns else []
        reg_drop = [c for c in (all_future + ['season_x', 'absolute_GW'] + non_predictive) if c not in keep]
        X_train_reg = train_i.drop(columns=reg_drop, errors='ignore')
        X_test_reg = test_fold.drop(columns=reg_drop, errors='ignore')

        y_train_pts = train_i[pts_col]
        y_train_min = (train_i[min_col] >= 60).astype(int)

        X_train_clf, X_test_clf = X_train_clf.align(X_test_clf, join='inner', axis=1, fill_value=0)
        X_train_reg, X_test_reg = X_train_reg.align(X_test_reg, join='inner', axis=1, fill_value=0)

        # Scale features
        scaler_clf = StandardScaler()
        num_clf = [c for c in X_train_clf.columns if X_train_clf[c].dtype not in ['uint8', 'bool']]
        X_train_clf[num_clf] = scaler_clf.fit_transform(X_train_clf[num_clf])
        X_test_clf[num_clf] = scaler_clf.transform(X_test_clf[num_clf])

        scaler_reg = StandardScaler()
        num_reg = [c for c in X_train_reg.columns if X_train_reg[c].dtype not in ['uint8', 'bool']]
        X_train_reg[num_reg] = scaler_reg.fit_transform(X_train_reg[num_reg])
        X_test_reg[num_reg] = scaler_reg.transform(X_test_reg[num_reg])

        # Sample weights
        latest = train_i['absolute_GW'].max()
        weights = WEIGHT_DECAY ** (latest - train_i['absolute_GW'])

        try:
            # --- Train Regressor ---
            rf_reg = RandomForestRegressor(
                random_state=42,
                n_jobs=1  # Set to 1 since we're already parallelizing at the task level
            )
            rs_reg = RandomizedSearchCV(
                rf_reg, param_distributions=PARAM_GRID_REG, n_iter=N_ITER_SEARCH,
                cv=CV_FOLDS, scoring='neg_mean_absolute_error',
                n_jobs=N_JOBS_PER_TASK,  # Use specified cores per task
                random_state=42, verbose=VERBOSE_LEVEL
            )
            rs_reg.fit(X_train_reg, y_train_pts, sample_weight=weights)
            best_reg = rs_reg.best_estimator_

            # --- Train Classifier ---
            rf_clf = RandomForestClassifier(
                random_state=42,
                n_jobs=1  # Set to 1 since we're already parallelizing at the task level
            )

            # Check for single class in target for stratification
            stratify_y = y_train_min if len(y_train_min.unique()) > 1 else None

            # Simplified calibration check
            if len(X_train_clf) < 10 or stratify_y is None:
                rs_clf = RandomizedSearchCV(
                    rf_clf, param_distributions=PARAM_GRID_CLF, n_iter=N_ITER_SEARCH,
                    cv=CV_FOLDS, scoring='roc_auc', n_jobs=N_JOBS_PER_TASK,
                    random_state=42, verbose=VERBOSE_LEVEL
                )
                rs_clf.fit(X_train_clf, y_train_min, sample_weight=weights)
                calibrated_clf = rs_clf.best_estimator_
            else:
                train_indices, calib_indices = train_test_split(
                    np.arange(len(X_train_clf)), test_size=0.2, random_state=42, stratify=stratify_y
                )
                X_train_inner, X_calib = X_train_clf.iloc[train_indices], X_train_clf.iloc[calib_indices]
                y_train_inner, y_calib = y_train_min.iloc[train_indices], y_train_min.iloc[calib_indices]
                weights_train = weights.iloc[train_indices]

                rs_clf = RandomizedSearchCV(
                    rf_clf, param_distributions=PARAM_GRID_CLF, n_iter=N_ITER_SEARCH,
                    cv=CV_FOLDS, scoring='roc_auc', n_jobs=N_JOBS_PER_TASK,
                    random_state=42, verbose=VERBOSE_LEVEL
                )
                rs_clf.fit(X_train_inner, y_train_inner, sample_weight=weights_train)

                calibrated_clf = CalibratedClassifierCV(rs_clf.best_estimator_, method='sigmoid', cv='prefit')
                calibrated_clf.fit(X_calib, y_calib)

            # --- Make Predictions ---
            final_points_pred = np.full(len(test_fold), np.nan)
            if not test_i.empty:
                test_indices = test_i.index
                X_test_clf_pred = X_test_clf.loc[test_indices]
                X_test_reg_pred = X_test_reg.loc[test_indices]

                minutes_proba = calibrated_clf.predict_proba(X_test_clf_pred)[:, 1]
                points_pred = best_reg.predict(X_test_reg_pred)

                # Map predictions back to the original test_fold index
                pred_series = pd.Series(np.where(minutes_proba >= 0.5, points_pred, 0), index=test_indices)
                final_points_pred = pred_series.reindex(test_fold.index).values

            gw_predictions[f'predicted_points_gw+{i}'] = final_points_pred
            gw_predictions[f'points_gw+{i}'] = test_fold[pts_col].values

        except Exception as e:
            logging.error(f"    ❌ Error in GW{gw}+{i}: {e}")
            logging.error(traceback.format_exc())
            gw_predictions[f'predicted_points_gw+{i}'] = np.full(len(test_fold), np.nan)
            gw_predictions[f'points_gw+{i}'] = test_fold[pts_col].values if pts_col in test_fold.columns else np.full(
                len(test_fold), np.nan)

    logging.info(f"✅ Completed task for GW {gw}")
    return pd.DataFrame(gw_predictions)


def main():
    """
    Main function to set up the Dask cluster, run parallel tasks, and save results.
    """
    try:
        # Create output directories
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        logging.info(f"✅ Models will be saved to: {MODEL_OUTPUT_DIR}")
        logging.info(f"💾 Predictions will be saved to: {os.path.join(PREDICTIONS_OUTPUT_DIR, PREDICTIONS_OUTPUT_FILE)}")

        # --- 🧱 DATA INGESTION & PREP ---
        if not os.path.exists(INPUT_DATA_FILE):
            logging.error(f"❌ Error: Input data file not found at '{INPUT_DATA_FILE}'")
            sys.exit(1)

        logging.info("📊 Loading data...")
        df = pd.read_csv(INPUT_DATA_FILE)
        df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])

        seasons = sorted(df['season_x'].unique(), key=lambda x: int(x.split('-')[0]))
        offsets = {s: sum(SEASON_LENGTHS.get(se, DEFAULT_GW_COUNT) for se in seasons[:i]) for i, s in
                   enumerate(seasons)}
        df['absolute_GW'] = df.apply(lambda row: offsets[row['season_x']] + row['GW'], axis=1)

        logging.info("🔄 Performing one-hot encoding...")
        categorical_cols = ['position', 'team', 'opponent_team', 'opponent_2_team', 'was_home',
                            'team_strength', 'opp_team_strength']
        df = pd.get_dummies(df, columns=[col for col in categorical_cols if col in df.columns], drop_first=True)
        logging.info("✅ Data preprocessing complete")

        ## PARALLEL-SPECIFIC CHANGE: Set up Dask cluster and client.
        # This creates a scheduler and workers to run the tasks.
        # LocalCUDACluster automatically finds the GPUs on the node.
        logging.info("🚀 Setting up Dask LocalCUDACluster for 4 GPUs...")
        # Assumes you have 4 GPUs visible to your job
        cluster = LocalCUDACluster(
            n_workers=4,
            threads_per_worker=(32 // 4)  # Distribute CPUs evenly among workers
        )
        client = Client(cluster)
        logging.info(f"✅ Dask dashboard link: {client.dashboard_link}")

        # --- 🎯 Walk-forward TRAINING & PREDICTION for 2024-25 ---
        season = '2024-25'
        all_predictions = []
        target_gws = sorted(df[df['season_x'] == season]['GW'].unique())
        logging.info(f"🎯 Target gameweeks for parallel processing: {target_gws}")

        # Share the dataframe with all workers once to avoid re-sending it.
        df_future = client.scatter(df, broadcast=True)

        # Create a dictionary of the configuration to pass to each worker
        config = {
            'PLANNING_HORIZON': PLANNING_HORIZON,
            'MODEL_OUTPUT_DIR': MODEL_OUTPUT_DIR,
            'N_ITER_SEARCH': N_ITER_SEARCH,
            'CV_FOLDS': CV_FOLDS,
            'WEIGHT_DECAY': WEIGHT_DECAY,
            'VERBOSE_LEVEL': VERBOSE_LEVEL,
            'N_JOBS_PER_TASK': N_JOBS_PER_TASK,
            'PARAM_GRID_REG': PARAM_GRID_REG,
            'PARAM_GRID_CLF': PARAM_GRID_CLF
        }

        ## PARALLEL-SPECIFIC CHANGE: Create a list of "delayed" tasks.
        # Dask builds a task graph without executing immediately.
        lazy_results = []
        for gw in target_gws:
            task = dask.delayed(process_gameweek)(gw, season, df_future, config)
            lazy_results.append(task)

        ## PARALLEL-SPECIFIC CHANGE: Execute all tasks in parallel.
        # dask.compute triggers the execution of the entire task graph.
        # Results will be a list of DataFrames (or None for failed tasks).
        logging.info(f"⏳ Computing {len(lazy_results)} tasks in parallel... This may take a while.")
        results = dask.compute(*lazy_results)

        # Filter out any None results from failed/skipped tasks
        all_predictions = [res for res in results if res is not None]

        # --- 💾 SAVE PREDICTIONS ---
        if all_predictions:
            logging.info(f"📊 Combining predictions from {len(all_predictions)} gameweeks...")
            final_predictions_df = pd.concat(all_predictions, ignore_index=True)

            base_cols = ['element', 'GW', 'name', 'value', 'team', 'position']
            pred_cols = sorted([c for c in final_predictions_df.columns if 'predicted_points' in c])
            actual_cols = sorted([c for c in final_predictions_df.columns if c.startswith('points_gw+')])

            final_predictions_df = final_predictions_df.reindex(columns=base_cols + pred_cols + actual_cols)

            output_path = os.path.join(PREDICTIONS_OUTPUT_DIR, PREDICTIONS_OUTPUT_FILE)
            final_predictions_df.to_csv(output_path, index=False)
            logging.info(f"✅ Predictions saved to: {output_path}")
            logging.info(f"📈 Final shape: {final_predictions_df.shape}")
        else:
            logging.warning("❌ No predictions were generated!")

        logging.info("\n🎉 Training and prediction complete!")
        client.close()
        cluster.close()

    except Exception as e:
        logging.error(f"❌ Fatal error in main execution: {e}")
        logging.error(traceback.format_exc())


if __name__ == "__main__":
    main()