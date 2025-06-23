import pandas as pd
from autogluon.tabular import TabularPredictor
import numpy as np
import os
import warnings
import sys
import traceback
import shutil
import logging

# Dask libraries for parallel execution
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import dask

# --- 📝 IMPROVED LOGGER CLASS ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_log_autogluon_parallel.txt", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

warnings.filterwarnings('ignore')

# --- ⚙️ CONFIGURATION ---
PLANNING_HORIZON = 6
MODEL_OUTPUT_DIR = 'ag_models_parallel'
PREDICTIONS_OUTPUT_DIR = '.'
PREDICTIONS_OUTPUT_FILE = 'predictions_autogluon_parallel.csv'
# File to store model performance data
MODEL_PERFORMANCE_FILE = 'model_performance_autogluon_parallel.csv'
INPUT_DATA_FILE = 'processed_fpl_data.csv'
WEIGHT_DECAY = 0.98
TIME_LIMIT_PER_MODEL = 300
# Using 'medium_quality' for stability, as it disables auto_stack and avoids Ray issues.
PRESET_QUALITY = 'high_quality'

SEASON_LENGTHS = {'2019-20': 47}
DEFAULT_GW_COUNT = 38


# --- ✨ AUTOGLUON FUNCTION FOR PARALLEL EXECUTION ---
def process_gameweek_ag(gw, season, df, config):
    """
    Trains models for a single gameweek and returns both predictions and model performance data.
    This function is designed to be run in parallel by a Dask worker.
    """
    for key, value in config.items():
        globals()[key] = value

    logging.info(f"---  Dask Task starting for GW {gw} ({season}) ---")

    train_fold = df[(df['season_x'] != season) | (df['GW'] < gw)]
    test_fold = df[(df['season_x'] == season) & (df['GW'] == gw)]

    if train_fold.empty or test_fold.empty:
        logging.warning(f"Skipping GW {gw}: insufficient data.")
        return None, None  # Return None for both dataframes

    gw_predictions = {
        'element': test_fold['element'].values,
        'GW': test_fold['GW'].values,
        'name': test_fold['name'].values,
        'value': test_fold['value'].values,
        'team': test_fold['team_x'].values,
        'position': test_fold['position'].values,
    }

    gw_leaderboards = []

    for i in range(1, PLANNING_HORIZON + 1):
        label_col = f'points_gw+{i}'
        logging.info(f"  [GW {gw}] Processing horizon +{i} (Target: {label_col})...")

        if label_col not in train_fold.columns or train_fold[label_col].isnull().all():
            gw_predictions[f'predicted_{label_col}'] = np.full(len(test_fold), np.nan)
            gw_predictions[label_col] = np.full(len(test_fold), np.nan)
            continue

        train_i = train_fold.dropna(subset=[label_col])

        if train_i.empty:
            continue

        all_future_pts = [f'points_gw+{j}' for j in range(1, PLANNING_HORIZON + 1)]
        all_future_mins = [f'minutes_gw+{j}' for j in range(1, PLANNING_HORIZON + 1)]
        non_predictive = ['name', 'element', 'kickoff_time', 'fixture', 'round',
                          'opp_team_name', 'team_x', 'understat_missing',
                          'opp_team_name_1', 'opp_team_name_2', 'season_x']

        cols_to_drop = [c for c in all_future_pts if c != label_col] + all_future_mins + non_predictive

        train_data = train_i.drop(columns=cols_to_drop, errors='ignore')
        test_data = test_fold.drop(columns=cols_to_drop, errors='ignore')
        train_cols = train_data.columns.drop(label_col)
        test_data = test_data[train_cols]

        latest_gw = train_data['absolute_GW'].max()
        weights = WEIGHT_DECAY ** (latest_gw - train_data['absolute_GW'])
        train_data['sample_weight'] = weights

        try:
            model_path = os.path.join(MODEL_OUTPUT_DIR, f'gw_{gw}_horizon_{i}')

            # Critical for parallel execution: Constrain AutoGluon to its worker's resources
            ag_resources = {'num_gpus': 1, 'num_cpus': 8}

            predictor = TabularPredictor(
                label=label_col,
                problem_type='regression',
                path=model_path,
                eval_metric='mean_absolute_error',
                sample_weight='sample_weight',
            )

            predictor.fit(
                train_data,
                presets=PRESET_QUALITY,
                time_limit=TIME_LIMIT_PER_MODEL,
                ag_args_fit={'resources': ag_resources},  # Pass resource constraints
                auto_stack=False
            )

            predictions = predictor.predict(test_data)
            leaderboard_df = predictor.leaderboard(silent=True)

            gw_predictions[f'predicted_{label_col}'] = np.maximum(0, predictions.values)
            gw_predictions[label_col] = test_fold[label_col].values

            leaderboard_df['GW'] = gw
            leaderboard_df['horizon'] = i
            gw_leaderboards.append(leaderboard_df)

            shutil.rmtree(model_path, ignore_errors=True)

        except Exception as e:
            logging.error(f"    ❌ Error in GW{gw}+{i} with AutoGluon: {e}")
            logging.error(traceback.format_exc())
            gw_predictions[f'predicted_{label_col}'] = np.full(len(test_fold), np.nan)
            gw_predictions[label_col] = test_fold.get(label_col, np.full(len(test_fold), np.nan))

    logging.info(f"✅ Dask Task completed for GW {gw}")

    final_leaderboard_df = pd.concat(gw_leaderboards, ignore_index=True) if gw_leaderboards else None
    return pd.DataFrame(gw_predictions), final_leaderboard_df


def main():
    """
    Main function to set up Dask, run parallel AutoGluon tasks, and save all results.
    """
    try:
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        logging.info(f"✅ Models will be saved to (and removed from): {MODEL_OUTPUT_DIR}")
        logging.info(f"💾 Predictions will be saved to: {os.path.join(PREDICTIONS_OUTPUT_DIR, PREDICTIONS_OUTPUT_FILE)}")
        logging.info(
            f"📊 Model performance data will be saved to: {os.path.join(PREDICTIONS_OUTPUT_DIR, MODEL_PERFORMANCE_FILE)}")

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
        logging.info("✅ Data loaded and prepped.")

        # --- DASK CLUSTER SETUP ---
        logging.info("🚀 Setting up Dask LocalCUDACluster for 4 GPUs...")
        cluster = LocalCUDACluster(n_workers=4, threads_per_worker=8)
        client = Client(cluster)
        logging.info(f"✅ Dask dashboard link: {client.dashboard_link}")

        season = '2024-25'
        target_gws = sorted(df[df['season_x'] == season]['GW'].unique())
        logging.info(f"🎯 Target gameweeks for parallel processing: {target_gws}")

        df_future = client.scatter(df, broadcast=True)

        config = {
            'PLANNING_HORIZON': PLANNING_HORIZON,
            'MODEL_OUTPUT_DIR': MODEL_OUTPUT_DIR,
            'WEIGHT_DECAY': WEIGHT_DECAY,
            'TIME_LIMIT_PER_MODEL': TIME_LIMIT_PER_MODEL,
            'PRESET_QUALITY': PRESET_QUALITY
        }

        lazy_results = [dask.delayed(process_gameweek_ag)(gw, season, df_future, config) for gw in target_gws]

        logging.info(f"⏳ Computing {len(lazy_results)} tasks in parallel with Dask...")
        results = dask.compute(*lazy_results)

        # --- PROCESS AND SAVE RESULTS ---
        all_prediction_dfs = []
        all_leaderboard_dfs = []

        for res in results:
            if res is not None:
                predictions_df, leaderboard_df = res
                if predictions_df is not None:
                    all_prediction_dfs.append(predictions_df)
                if leaderboard_df is not None:
                    all_leaderboard_dfs.append(leaderboard_df)

        # Save predictions
        if all_prediction_dfs:
            final_predictions_df = pd.concat(all_prediction_dfs, ignore_index=True)
            base_cols = ['element', 'GW', 'name', 'value', 'team', 'position']
            pred_cols = sorted([c for c in final_predictions_df.columns if c.startswith('predicted_')])
            actual_cols = sorted([c for c in final_predictions_df.columns if c.startswith('points_gw+')])
            final_predictions_df = final_predictions_df.reindex(columns=base_cols + pred_cols + actual_cols)
            final_predictions_df.to_csv(os.path.join(PREDICTIONS_OUTPUT_DIR, PREDICTIONS_OUTPUT_FILE), index=False)
            logging.info(f"✅ Predictions saved. Final shape: {final_predictions_df.shape}")
        else:
            logging.warning("❌ No predictions were generated!")

        # Save model performance data
        if all_leaderboard_dfs:
            final_leaderboard_df = pd.concat(all_leaderboard_dfs, ignore_index=True)
            final_leaderboard_df.to_csv(os.path.join(PREDICTIONS_OUTPUT_DIR, MODEL_PERFORMANCE_FILE), index=False)
            logging.info(f"✅ Model performance data saved. Final shape: {final_leaderboard_df.shape}")
        else:
            logging.warning("❌ No model performance data was generated!")

        logging.info("\n🎉 AutoGluon parallel training and prediction complete!")

    except Exception as e:
        logging.error(f"❌ Fatal error in main execution: {e}")
        logging.error(traceback.format_exc())
    finally:
        if 'client' in locals():
            client.close()
        if 'cluster' in locals():
            cluster.close()


if __name__ == "__main__":
    main()