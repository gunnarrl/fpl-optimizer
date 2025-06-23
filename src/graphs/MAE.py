import pandas as pd
from sklearn.metrics import mean_absolute_error


def calculate_combined_mae(df: pd.DataFrame):
    """
    Calculates and prints the combined Mean Absolute Error (MAE) for all players
    for each forecast horizon from GW+1 to GW+6.

    Args:
        df (pd.DataFrame): The DataFrame containing player prediction data.
                           The file should contain all players, not be pre-filtered.
    """
    print("--- Combined MAE for All Players ---")

    # Loop through each forecast horizon (GW+1 to GW+6)
    for i in range(1, 7):
        # Define the column names for the current forecast horizon
        pred_col = f'predicted_points_gw+{i}'
        actual_col = f'points_gw+{i}'

        # Verify that the necessary columns exist in the DataFrame
        if pred_col not in df.columns or actual_col not in df.columns:
            print(f"Data for GW+{i} not available. Skipping.")
            continue

        # --- MAE CALCULATION FOR ALL PLAYERS ---
        # Create a temporary DataFrame with only the prediction and actual columns
        # and drop all rows where either value is missing to ensure a fair comparison.
        mae_data = df[[pred_col, actual_col]].dropna()

        mae_score = 0
        if not mae_data.empty:
            # Calculate the MAE using all available data for this horizon
            mae_score = mean_absolute_error(mae_data[actual_col], mae_data[pred_col])
            print(f"MAE for Forecast Horizon GW+{i}: {mae_score:.4f}")
        else:
            print(f"No valid data pairs found for GW+{i} to calculate MAE.")

    print("------------------------------------")


# --- How to use this function ---

# 1. Load your complete, unfiltered CSV into a pandas DataFrame.
#    Make sure the path to your CSV file is correct.
try:
    # Use the same path as the previous script for consistency
    df_full = pd.read_csv('../../data/output/predictions/koa_predictions_updated.csv')

    # 2. Call the function to calculate and print the MAEs.
    calculate_combined_mae(df_full)

except FileNotFoundError:
    print("Error: 'koa_predictions_updated.csv' not found.")
    print("Please make sure the CSV file is in the correct directory, or provide the full path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")