import pandas as pd
from autogluon.tabular import TabularPredictor
import numpy as np
import os
import warnings
import sys
import traceback
import shutil

# Dask libraries for parallel execution
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
import dask
import logging

# --- 📝 IMPROVED LOGGER CLASS ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_log_autogluon.txt", mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)

warnings.filterwarnings('ignore')

# --- ⚙️ CONFIGURATION ---
PLANNING_HORIZON = 6
MODEL_OUTPUT_DIR = 'ag_models'
PREDICTIONS_OUTPUT_DIR = '.'
PREDICTIONS_OUTPUT_FILE = 'predictions_autogluon.csv'
INPUT_DATA_FILE = 'processed_fpl_data.csv'
WEIGHT_DECAY = 0.98
# AutoGluon-specific: Time limit in seconds for fitting each model (per horizon, per gw)
TIME_LIMIT_PER_MODEL = 300
# Set to 'best_quality' for better models, 'medium_quality' for faster runs
PRESET_QUALITY = 'high_quality'

SEASON_LENGTHS = {'2019-20': 47}
DEFAULT_GW_COUNT = 38


# --- ✨ AUTOGLUON CONVERSION ---
# This function is now designed to be run in parallel by Dask, using AutoGluon for modeling.
def process_gameweek_ag(gw, season, df, config):
    """
    Trains AutoGluon models and generates predictions for a single gameweek.
    """
    # Unpack config dictionary
    for key, value in config.items():
        globals()[key] = value

    logging.info(f"---  AutoML Task starting for GW {gw} ({season}) ---")

    # Split data: train on past, test on current gameweek
    train_fold = df[(df['season_x'] != season) | (df['GW'] < gw)]
    test_fold = df[(df['season_x'] == season) & (df['GW'] == gw)]

    if train_fold.empty or test_fold.empty:
        logging.warning(f"Skipping GW {gw}: insufficient data.")
        return None

    # This dictionary will hold the results for the current gameweek
    gw_predictions = {
        'element': test_fold['element'].values,
        'GW': test_fold['GW'].values,
        'name': test_fold['name'].values,
        'value': test_fold['value'].values,
        'team': test_fold['team_x'].values,
        'position': test_fold['position'].values,
    }

    # Train a separate model for each future gameweek in the horizon
    for i in range(1, PLANNING_HORIZON + 1):
        label_col = f'points_gw+{i}'
        logging.info(f"  [GW {gw}] Processing horizon +{i} (Target: {label_col})...")

        # Ensure target columns exist for training and evaluation
        if label_col not in train_fold.columns:
            gw_predictions[f'predicted_{label_col}'] = np.full(len(test_fold), np.nan)
            gw_predictions[label_col] = np.full(len(test_fold), np.nan)
            continue

        train_i = train_fold.dropna(subset=[label_col])

        if train_i.empty:
            logging.warning(f"    [GW {gw}] No training data for +{i}, skipping...")
            gw_predictions[f'predicted_{label_col}'] = np.full(len(test_fold), np.nan)
            gw_predictions[label_col] = test_fold[label_col].values
            continue

        # --- DATA PREP FOR AUTOGLUON ---
        # AutoGluon handles feature scaling and encoding internally. We just need to remove leaky features.
        all_future_pts = [f'points_gw+{j}' for j in range(1, PLANNING_HORIZON + 1)]
        all_future_mins = [f'minutes_gw+{j}' for j in range(1, PLANNING_HORIZON + 1)]
        non_predictive = ['name', 'element', 'kickoff_time', 'fixture', 'round',
                          'opp_team_name', 'team_x', 'understat_missing',
                          'opp_team_name_1', 'opp_team_name_2', 'season_x']

        # Columns to be dropped from the training data
        # We keep the target column (`label_col`) but drop all other future point/minute columns
        cols_to_drop = [c for c in all_future_pts if c != label_col] + all_future_mins + non_predictive

        train_data = train_i.drop(columns=cols_to_drop, errors='ignore')
        test_data = test_fold.drop(columns=cols_to_drop, errors='ignore')

        # Align columns just in case - crucial for prediction
        train_cols = train_data.columns.drop(label_col)
        test_data = test_data[train_cols]

        # Calculate sample weights to prioritize recent data
        latest_gw = train_data['absolute_GW'].max()
        weights = WEIGHT_DECAY ** (latest_gw - train_data['absolute_GW'])
        train_data['sample_weight'] = weights

        try:
            # --- TRAIN WITH AUTOGLUON ---
            # Define a unique path for each model to avoid conflicts during parallel runs
            model_path = os.path.join(MODEL_OUTPUT_DIR, f'gw_{gw}_horizon_{i}')

            # Resources available to this Dask worker (1 GPU, 8 CPUs)
            ag_resources = {'num_gpus': 1, 'num_cpus': 8}

            predictor = TabularPredictor(
                label=label_col,
                path=model_path,
                eval_metric='mean_absolute_error',
            )

            predictor.fit(
                train_data,
                presets=PRESET_QUALITY,
                time_limit=TIME_LIMIT_PER_MODEL,
                ag_args_fit={'resources': ag_resources},
                sample_weight='sample_weight'  # Use the column name
            )

            # --- PREDICT ---
            predictions = predictor.predict(test_data)

            # Clip predictions at 0, as negative points are rare/impossible
            gw_predictions[f'predicted_{label_col}'] = np.maximum(0, predictions.values)
            gw_predictions[label_col] = test_fold[label_col].values

            # Optional: Clean up disk space
            shutil.rmtree(model_path, ignore_errors=True)

        except Exception as e:
            logging.error(f"    ❌ Error in GW{gw}+{i} with AutoGluon: {e}")
            logging.error(traceback.format_exc())
            gw_predictions[f'predicted_{label_col}'] = np.full(len(test_fold), np.nan)
            gw_predictions[label_col] = test_fold[label_col].values if label_col in test_fold.columns else np.full(
                len(test_fold), np.nan)

    logging.info(f"✅ AutoML Task completed for GW {gw}")
    return pd.DataFrame(gw_predictions)


def main():
    """
    Main function to set up Dask, run parallel AutoGluon tasks, and save results.
    """
    try:
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        logging.info(f"✅ AutoGluon models will be saved to (and removed from): {MODEL_OUTPUT_DIR}")
        logging.info(f"💾 Predictions will be saved to: {os.path.join(PREDICTIONS_OUTPUT_DIR, PREDICTIONS_OUTPUT_FILE)}")

        if not os.path.exists(INPUT_DATA_FILE):
            logging.error(f"❌ Error: Input data file not found at '{INPUT_DATA_FILE}'")
            sys.exit(1)

        logging.info("📊 Loading data...")
        df = pd.read_csv(INPUT_DATA_FILE)
        df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])

        # Calculate a continuous gameweek counter across seasons
        seasons = sorted(df['season_x'].unique(), key=lambda x: int(x.split('-')[0]))
        offsets = {s: sum(SEASON_LENGTHS.get(se, DEFAULT_GW_COUNT) for se in seasons[:i]) for i, s in
                   enumerate(seasons)}
        df['absolute_GW'] = df.apply(lambda row: offsets[row['season_x']] + row['GW'], axis=1)
        logging.info("✅ Data loaded and prepped.")

        # --- DASK CLUSTER SETUP ---
        logging.info("🚀 Setting up Dask LocalCUDACluster for 4 GPUs...")
        cluster = LocalCUDACluster(
            n_workers=4,
            threads_per_worker=8  # 32 CPUs / 4 workers = 8 CPUs per worker
        )
        client = Client(cluster)
        logging.info(f"✅ Dask dashboard link: {client.dashboard_link}")

        # --- WALK-FORWARD TRAINING WITH AUTOGLUON ---
        season = '2024-25'
        target_gws = sorted(df[df['season_x'] == season]['GW'].unique())
        logging.info(f"🎯 Target gameweeks for parallel processing: {target_gws}")

        # Scatter the dataframe to all workers once
        df_future = client.scatter(df, broadcast=True)

        # Configuration to pass to each parallel task
        config = {
            'PLANNING_HORIZON': PLANNING_HORIZON,
            'MODEL_OUTPUT_DIR': MODEL_OUTPUT_DIR,
            'WEIGHT_DECAY': WEIGHT_DECAY,
            'TIME_LIMIT_PER_MODEL': TIME_LIMIT_PER_MODEL,
            'PRESET_QUALITY': PRESET_QUALITY
        }

        # Create a list of "delayed" Dask tasks
        lazy_results = [
            dask.delayed(process_gameweek_ag)(gw, season, df_future, config) for gw in target_gws
        ]

        # Execute all tasks in parallel
        logging.info(f"⏳ Computing {len(lazy_results)} tasks in parallel with AutoGluon...")
        results = dask.compute(*lazy_results)

        # Combine results
        all_predictions = [res for res in results if res is not None]
        if all_predictions:
            final_df = pd.concat(all_predictions, ignore_index=True)

            # Reorder columns for clarity
            base_cols = ['element', 'GW', 'name', 'value', 'team', 'position']
            pred_cols = sorted([c for c in final_df.columns if c.startswith('predicted_')])
            actual_cols = sorted([c for c in final_df.columns if c.startswith('points_gw+')])
            final_df = final_df.reindex(columns=base_cols + pred_cols + actual_cols)

            output_path = os.path.join(PREDICTIONS_OUTPUT_DIR, PREDICTIONS_OUTPUT_FILE)
            final_df.to_csv(output_path, index=False)
            logging.info(f"✅ Predictions saved to: {output_path}")
            logging.info(f"📈 Final shape: {final_df.shape}")
        else:
            logging.warning("❌ No predictions were generated!")

        logging.info("\n🎉 AutoGluon training and prediction complete!")

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