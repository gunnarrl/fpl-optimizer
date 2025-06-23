import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, roc_auc_score
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
WEIGHT_DECAY = 0.98
VERBOSE_LEVEL = 0  # Quieter logs for parallel runs

# Neural Network Configuration
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 10  # Early stopping patience
HIDDEN_SIZES = [256, 128, 64]  # Hidden layer sizes
DROPOUT_RATE = 0.3

SEASON_LENGTHS = {'2019-20': 47}
DEFAULT_GW_COUNT = 38

# Understat columns that need special handling
UNDERSTAT_COLS = ['xG', 'xA', 'key_passes', 'npg', 'npxG', 'xGChain', 'xGBuildup', 'xP']


class FPLDataset(Dataset):
    """Custom PyTorch dataset for FPL data"""

    def __init__(self, X, y, weights=None):
        self.X = torch.FloatTensor(X.values if hasattr(X, 'values') else X)
        self.y = torch.FloatTensor(y.values if hasattr(y, 'values') else y)
        self.weights = torch.FloatTensor(
            weights.values if hasattr(weights, 'values') else weights) if weights is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.weights is not None:
            return self.X[idx], self.y[idx], self.weights[idx]
        return self.X[idx], self.y[idx]


class PointsRegressor(nn.Module):
    """Neural network for points regression"""

    def __init__(self, input_size, hidden_sizes=None, dropout_rate=0.3):
        super(PointsRegressor, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


class MinutesClassifier(nn.Module):
    """Neural network for minutes classification (>=60 minutes)"""

    def __init__(self, input_size, hidden_sizes=None, dropout_rate=0.3):
        super(MinutesClassifier, self).__init__()

        if hidden_sizes is None:
            hidden_sizes = [256, 128, 64]

        layers = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size

        # Output layer with sigmoid
        layers.extend([
            nn.Linear(prev_size, 1),
            nn.Sigmoid()
        ])

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze()


def clean_missing_data(df):
    """
    Clean missing data according to specified rules:
    1. Fill gw+{i} columns with 0
    2. Fill understat columns with 0 if minutes == 0, otherwise with per_90 * minutes / 90
    3. Everything else should already be filled
    """
    logging.info("🧹 Cleaning missing data...")

    # Fill gw+{i} columns with 0
    gw_cols = [col for col in df.columns if 'gw+' in col]
    for col in gw_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            logging.info(f"  Filled {col} missing values with 0")

    # Handle understat columns
    for col in UNDERSTAT_COLS:
        if col in df.columns:
            # First, fill with 0 where minutes == 0
            zero_minutes_mask = (df['minutes'] == 0) | (df['minutes'].isna())
            df.loc[zero_minutes_mask, col] = df.loc[zero_minutes_mask, col].fillna(0)

            # For non-zero minutes, use per_90 version if available
            per_90_col = f'per_90_{col}'
            if per_90_col in df.columns:
                non_zero_mask = ~zero_minutes_mask & df[col].isna()
                if non_zero_mask.any():
                    df.loc[non_zero_mask, col] = (
                            df.loc[non_zero_mask, per_90_col] *
                            df.loc[non_zero_mask, 'minutes'] / 90
                    )
                    logging.info(f"  Filled {col} using {per_90_col} for non-zero minutes")

            # Fill any remaining missing values with 0
            df[col] = df[col].fillna(0)

    logging.info("✅ Data cleaning complete")
    return df


def train_neural_network(model, train_loader, val_loader, criterion, optimizer, epochs, patience, device):
    """Train neural network with early stopping"""
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_samples = 0

        for batch in train_loader:
            if len(batch) == 3:  # With weights
                X_batch, y_batch, weights_batch = batch
                X_batch, y_batch, weights_batch = X_batch.to(device), y_batch.to(device), weights_batch.to(device)
            else:  # Without weights
                X_batch, y_batch = batch
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                weights_batch = None

            optimizer.zero_grad()
            outputs = model(X_batch)

            if weights_batch is not None:
                loss = (criterion(outputs, y_batch) * weights_batch).mean()
            else:
                loss = criterion(outputs, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(X_batch)
            train_samples += len(X_batch)

        train_loss /= train_samples
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    X_batch, y_batch, _ = batch
                else:
                    X_batch, y_batch = batch
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * len(X_batch)
                val_samples += len(X_batch)

        val_loss /= val_samples
        val_losses.append(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if epoch % 20 == 0 or epoch == epochs - 1:
            logging.info(f"    Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if patience_counter >= patience:
            logging.info(f"    Early stopping at epoch {epoch}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, train_losses, val_losses


## PARALLEL-SPECIFIC CHANGE: Encapsulate the logic for a single gameweek into a function.
def process_gameweek(gw, season, df, config):
    """
    Trains models and generates predictions for a single gameweek using neural networks.
    This function is designed to be run in parallel by Dask.
    """
    # Unpack config dictionary
    for key, value in config.items():
        globals()[key] = value

    logging.info(f"--- Starting task for GW {gw} ({season}) ---")

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"  Using device: {device}")

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
            # --- Train Regressor (Neural Network) ---
            logging.info(f"    [GW {gw}] Training points regressor...")

            # Split training data for validation
            X_train_reg_split, X_val_reg, y_train_pts_split, y_val_pts, weights_train, weights_val = train_test_split(
                X_train_reg, y_train_pts, weights, test_size=0.2, random_state=42
            )

            # Create datasets and data loaders
            train_dataset_reg = FPLDataset(X_train_reg_split, y_train_pts_split, weights_train)
            val_dataset_reg = FPLDataset(X_val_reg, y_val_pts)

            train_loader_reg = DataLoader(train_dataset_reg, batch_size=BATCH_SIZE, shuffle=True)
            val_loader_reg = DataLoader(val_dataset_reg, batch_size=BATCH_SIZE, shuffle=False)

            # Initialize model
            regressor = PointsRegressor(
                input_size=X_train_reg.shape[1],
                hidden_sizes=HIDDEN_SIZES,
                dropout_rate=DROPOUT_RATE
            ).to(device)

            # Loss and optimizer
            criterion_reg = nn.MSELoss(reduction='none')
            optimizer_reg = optim.Adam(regressor.parameters(), lr=LEARNING_RATE)

            # Train the model
            regressor, _, _ = train_neural_network(
                regressor, train_loader_reg, val_loader_reg, criterion_reg, optimizer_reg,
                EPOCHS, PATIENCE, device
            )

            # --- Train Classifier (Neural Network) ---
            logging.info(f"    [GW {gw}] Training minutes classifier...")

            # Check for single class in target
            if len(y_train_min.unique()) <= 1:
                logging.warning(f"    [GW {gw}] Single class in minutes target, using dummy classifier")
                minutes_proba = np.full(len(test_fold), 0.5)
            else:
                # Split training data for validation
                X_train_clf_split, X_val_clf, y_train_min_split, y_val_min, weights_train_clf, weights_val_clf = train_test_split(
                    X_train_clf, y_train_min, weights, test_size=0.2, random_state=42, stratify=y_train_min
                )

                # Create datasets and data loaders
                train_dataset_clf = FPLDataset(X_train_clf_split, y_train_min_split, weights_train_clf)
                val_dataset_clf = FPLDataset(X_val_clf, y_val_min)

                train_loader_clf = DataLoader(train_dataset_clf, batch_size=BATCH_SIZE, shuffle=True)
                val_loader_clf = DataLoader(val_dataset_clf, batch_size=BATCH_SIZE, shuffle=False)

                # Initialize model
                classifier = MinutesClassifier(
                    input_size=X_train_clf.shape[1],
                    hidden_sizes=HIDDEN_SIZES,
                    dropout_rate=DROPOUT_RATE
                ).to(device)

                # Loss and optimizer
                criterion_clf = nn.BCELoss(reduction='none')
                optimizer_clf = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)

                # Train the model
                classifier, _, _ = train_neural_network(
                    classifier, train_loader_clf, val_loader_clf, criterion_clf, optimizer_clf,
                    EPOCHS, PATIENCE, device
                )

                # Get probabilities for test set
                classifier.eval()
                with torch.no_grad():
                    if not test_i.empty:
                        test_indices = test_i.index
                        X_test_clf_pred = X_test_clf.loc[test_indices]
                        X_test_tensor = torch.FloatTensor(X_test_clf_pred.values).to(device)
                        minutes_proba_tensor = classifier(X_test_tensor)
                        minutes_proba_values = minutes_proba_tensor.cpu().numpy()

                        # Map back to full test set
                        minutes_proba = np.full(len(test_fold), 0.5)
                        test_idx_mapping = {idx: i for i, idx in enumerate(test_indices)}
                        for i, idx in enumerate(test_fold.index):
                            if idx in test_idx_mapping:
                                minutes_proba[i] = minutes_proba_values[test_idx_mapping[idx]]
                    else:
                        minutes_proba = np.full(len(test_fold), 0.5)

            # --- Make Predictions ---
            final_points_pred = np.full(len(test_fold), np.nan)
            if not test_i.empty:
                test_indices = test_i.index
                X_test_reg_pred = X_test_reg.loc[test_indices]

                regressor.eval()
                with torch.no_grad():
                    X_test_tensor = torch.FloatTensor(X_test_reg_pred.values).to(device)
                    points_pred_tensor = regressor(X_test_tensor)
                    points_pred_values = points_pred_tensor.cpu().numpy()

                # Map predictions back to the original test_fold index
                test_idx_mapping = {idx: i for i, idx in enumerate(test_indices)}
                for i, idx in enumerate(test_fold.index):
                    if idx in test_idx_mapping:
                        pred_idx = test_idx_mapping[idx]
                        if minutes_proba[i] >= 0.5:
                            final_points_pred[i] = points_pred_values[pred_idx]
                        else:
                            final_points_pred[i] = 0

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

        # Clean missing data according to specified rules
        df = clean_missing_data(df)

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
        logging.info("🚀 Setting up Dask LocalCUDACluster for 4 GPUs...")
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
            'WEIGHT_DECAY': WEIGHT_DECAY,
            'VERBOSE_LEVEL': VERBOSE_LEVEL,
            'BATCH_SIZE': BATCH_SIZE,
            'LEARNING_RATE': LEARNING_RATE,
            'EPOCHS': EPOCHS,
            'PATIENCE': PATIENCE,
            'HIDDEN_SIZES': HIDDEN_SIZES,
            'DROPOUT_RATE': DROPOUT_RATE
        }

        ## PARALLEL-SPECIFIC CHANGE: Create a list of "delayed" tasks.
        lazy_results = []
        for gw in target_gws:
            task = dask.delayed(process_gameweek)(gw, season, df_future, config)
            lazy_results.append(task)

        ## PARALLEL-SPECIFIC CHANGE: Execute all tasks in parallel.
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