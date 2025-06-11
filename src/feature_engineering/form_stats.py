import pandas as pd
import numpy as np
import os


def calculate_form_stats(input_filepath, output_filepath):
    """
    Loads player data with aggregates and calculates rolling 3 and 5-gameweek form statistics.

    This script handles Understat columns conditionally, excluding gameweeks from the average
    where data is marked as missing.

    Args:
        input_filepath (str): Path to the master CSV file with aggregates (e.g., master_with_aggregates.csv).
        output_filepath (str): Path to save the final CSV with form columns.
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

    # Check for required columns
    if 'understat_missing' not in df.columns:
        print("Error: 'understat_missing' column not found. This is required for conditional form calculations.")
        return

    if 'kickoff_time' in df.columns:
        df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    else:
        print("Error: 'kickoff_time' column not found, which is required for sorting. Aborting.")
        return

    # --- 2. Sort Data for Chronological Rolling Calculations ---
    df.sort_values(by=['season_x', 'element', 'kickoff_time'], inplace=True)
    print("Data sorted by season, player (element), and kickoff_time.")

    # --- 3. Define Columns and Windows for Form Calculation ---
    stats_to_process = [
        'xP', 'goals_scored', 'saves', 'penalties_saved', 'assists', 'xG', 'xA',
        'total_points', 'minutes', 'key_passes', 'npg', 'npxG', 'xGChain',
        'xGBuildup', 'own_goals', 'penalties_missed', 'clean_sheets', 'bps',
        'threat', 'ict_index', 'influence', 'creativity'
    ]

    understat_related_stats = [
        'xG', 'xA', 'key_passes', 'npg', 'npxG', 'xGChain', 'xGBuildup'
    ]

    # Define the rolling windows for form calculation
    rolling_windows = [3, 5]
    all_form_cols = []

    # Filter to only use columns that exist in the dataframe
    verified_stats = [col for col in stats_to_process if col in df.columns]

    # --- 4. Calculate Form using rolling windows ---
    # The calculation is grouped by each player within each season for each window size
    for window in rolling_windows:
        print(f"\n--- Calculating {window}-Gameweek Rolling Form Statistics ---")
        for col in verified_stats:
            form_col_name = f'form_{window}_{col}'
            all_form_cols.append(form_col_name)

            if col in understat_related_stats:
                # Conditional calculation for Understat stats
                temp_stat_col = f'temp_{col}'
                df[temp_stat_col] = np.where(df['understat_missing'] == 1, np.nan, df[col])

                # Calculate rolling mean, which ignores NaNs and adjusts the denominator
                rolling_avg = df.groupby(['season_x', 'element'])[temp_stat_col].rolling(window=window,
                                                                                         min_periods=1).mean()

                df[form_col_name] = rolling_avg.reset_index(level=[0, 1], drop=True)
                df.drop(columns=[temp_stat_col], inplace=True)
                print(f"  > Created '{form_col_name}' (conditional rolling average).")

            else:
                # Standard rolling average for non-Understat columns
                rolling_avg = df.groupby(['season_x', 'element'])[col].rolling(window=window, min_periods=1).mean()
                df[form_col_name] = rolling_avg.reset_index(level=[0, 1], drop=True)
                print(f"  > Created '{form_col_name}' (standard rolling average).")

    # Fill any potential NaNs created by the rolling operations with 0
    df[all_form_cols] = df[all_form_cols].fillna(0)
    print("\nFilled all NaN values in new form columns with 0.")

    # --- 5. Save the Final Dataset ---
    try:
        output_dir = os.path.dirname(output_filepath)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df.to_csv(output_filepath, index=False)
        print(f"\nSuccessfully saved the final dataset with form statistics to '{output_filepath}'.")
        print(f"Final dataset shape: {df.shape}")
    except Exception as e:
        print(f"\nError: Could not save the final file. Reason: {e}")


if __name__ == '__main__':
    # Define the input and output filepaths
    input_file = '../../data/processed/master_with_aggregates.csv'
    output_file_with_form = '../../data/processed/master_with_form.csv'

    # Run the function
    calculate_form_stats(input_file, output_file_with_form)