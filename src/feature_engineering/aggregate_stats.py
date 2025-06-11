import pandas as pd
import numpy as np
import os


def calculate_aggregate_stats(input_filepath, output_filepath):
    """
    Loads player data, calculates season-to-date aggregate statistics for each player
    on a gameweek-by-gameweek basis, and saves the enhanced data to a new file.
    This version specifically excludes Understat data from aggregation if it's marked as missing.
    It also adds per 90 minute statistics for the aggregated columns.

    Args:
        input_filepath (str): Path to the master CSV file (e.g., master_combined_with_xp.csv).
        output_filepath (str): Path to save the final CSV with aggregate columns.
    """
    # --- 1. Load the Dataset ---
    try:
        df = pd.read_csv(input_filepath)
        print(f"Successfully loaded dataset from '{input_filepath}'. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: The input file was not found at '{input_filepath}'. Please run the previous scripts first.")
        return
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return

    # Check if 'understat_missing' column exists, as it's essential for the logic
    if 'understat_missing' not in df.columns:
        print(
            "Error: 'understat_missing' column not found in the input file. This column is required for the conditional aggregation logic. Aborting.")
        return

    # It's also a good idea to ensure 'kickoff_time' is a datetime object for robust sorting
    if 'kickoff_time' in df.columns:
        df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    else:
        print("Error: 'kickoff_time' column not found, which is required for sorting. Aborting.")
        return

    # --- 2. Sort Data for Chronological Aggregation ---
    # FIX: Sorting by 'kickoff_time' instead of 'GW' for accurate chronological order.
    df.sort_values(by=['season_x', 'element', 'kickoff_time'], inplace=True)
    print("Data sorted by season, player (element), and kickoff_time.")

    # --- 3. Define Columns for Aggregation (Separating Understat from others) ---
    understat_stats_to_aggregate = [
        'xG', 'xA', 'key_passes', 'npg', 'npxG', 'xGChain', 'xGBuildup'
    ]

    other_stats_to_aggregate = [
        'xP', 'goals_scored', 'saves', 'penalties_saved', 'assists',
        'total_points', 'minutes', 'own_goals', 'penalties_missed', 'clean_sheets', 'threat', 'ict_index', 'influence', 'creativity'
    ]

    # Verify which of the requested columns actually exist in the DataFrame
    verified_understat_cols = [col for col in understat_stats_to_aggregate if col in df.columns]
    verified_other_cols = [col for col in other_stats_to_aggregate if col in df.columns]

    all_requested = understat_stats_to_aggregate + other_stats_to_aggregate
    all_verified = verified_understat_cols + verified_other_cols
    missing_cols = [col for col in all_requested if col not in all_verified]
    if missing_cols:
        print(f"Warning: The following columns were not found and will be skipped: {missing_cols}")

    if not all_verified:
        print("Error: None of the specified columns for aggregation exist in the input file. Aborting.")
        return

    print("\n--- Calculating Aggregates ---")

    # --- 4. Calculate Cumulative Sums with Conditional Logic for Understat columns ---
    print("\nProcessing stats with 'understat_missing' condition:")
    for col in verified_understat_cols:
        aggregate_col_name = f'aggregate_{col}'
        temp_col_name = f'temp_for_sum_{col}'
        df[temp_col_name] = np.where(df['understat_missing'] == 1, 0, df[col])
        df[aggregate_col_name] = df.groupby(['season_x', 'element'])[temp_col_name].cumsum()
        df.drop(columns=[temp_col_name], inplace=True)
        print(f"  > Created '{aggregate_col_name}' column (conditional).")

    # --- 5. Calculate Standard Cumulative Sums for Other Stats ---
    print("\nProcessing standard stats:")
    for col in verified_other_cols:
        aggregate_col_name = f'aggregate_{col}'
        df[aggregate_col_name] = df.groupby(['season_x', 'element'])[col].cumsum()
        print(f"  > Created '{aggregate_col_name}' column.")

    # --- 6. Calculate "Per 90 Minutes" Statistics ---
    print("\n--- Calculating Per 90 Minute Statistics ---")
    if 'aggregate_minutes' in df.columns:
        # Calculate the number of 90-minute periods played
        df['num_90s'] = df['aggregate_minutes'] / 90.0

        # Define all columns that have been aggregated (excluding minutes itself)
        stats_for_per_90 = verified_understat_cols + [col for col in verified_other_cols if col != 'minutes']

        for col in stats_for_per_90:
            aggregate_col_name = f'aggregate_{col}'
            per_90_col_name = f'per_90_{col}'

            # Calculate the per-90 stat, handling cases with 0 minutes to avoid division by zero
            df[per_90_col_name] = np.where(
                df['num_90s'] > 0,
                df[aggregate_col_name] / df['num_90s'],
                0
            )
            print(f"  > Created '{per_90_col_name}' column.")

        # Clean up the intermediate 'num_90s' column
        df.drop(columns=['num_90s'], inplace=True)
    else:
        print("Warning: 'aggregate_minutes' column not found. Cannot calculate per 90 stats.")

    # --- 7. Save the Final Dataset ---
    try:
        output_dir = os.path.dirname(output_filepath)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df.to_csv(output_filepath, index=False)
        print(f"\nSuccessfully saved the final dataset with aggregates to '{output_filepath}'.")
        print(f"Final dataset shape: {df.shape}")
    except Exception as e:
        print(f"\nError: Could not save the final file. Reason: {e}")


if __name__ == '__main__':
    # Define the input and output filepaths
    input_file = '../../data/raw/cleaned/master_combined_with_xp.csv'
    output_file_with_aggregates = '../../data/processed/master_with_aggregates.csv'

    # Run the function
    calculate_aggregate_stats(input_file, output_file_with_aggregates)