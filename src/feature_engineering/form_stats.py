import pandas as pd
import numpy as np
import os


def calculate_form_stats(input_filepath, output_filepath):
    """
    Loads player data with aggregates and calculates rolling 5-gameweek form statistics.

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

    # It's also a good idea to ensure 'kickoff_time' is a datetime object for robust sorting
    if 'kickoff_time' in df.columns:
        df['kickoff_time'] = pd.to_datetime(df['kickoff_time'])
    else:
        print("Error: 'kickoff_time' column not found, which is required for sorting. Aborting.")
        return

    # --- 2. Sort Data for Chronological Rolling Calculations ---
    # FIX: Sorting by 'kickoff_time' instead of 'GW' for accurate chronological order.
    df.sort_values(by=['season_x', 'element', 'kickoff_time'], inplace=True)
    print("Data sorted by season, player (element), and kickoff_time.")

    # --- 3. Define Columns for Form Calculation ---
    stats_to_process = [
        'xP', 'goals_scored', 'saves', 'penalties_saved', 'assists', 'xG', 'xA',
        'total_points', 'minutes', 'key_passes', 'npg', 'npxG', 'xGChain',
        'xGBuildup', 'own_goals', 'penalties_missed', 'clean_sheets', 'bps'
    ]

    understat_related_stats = [
        'xG', 'xA', 'key_passes', 'npg', 'npxG', 'xGChain', 'xGBuildup'
    ]

    # Filter to only use columns that exist in the dataframe
    verified_stats = [col for col in stats_to_process if col in df.columns]

    print("\n--- Calculating 5-Gameweek Rolling Form Statistics ---")

    # --- 4. Calculate Form using a rolling window ---
    # The calculation is grouped by each player within each season
    for col in verified_stats:
        form_col_name = f'form_{col}'

        # The logic differs for Understat columns vs others
        if col in understat_related_stats:
            # Conditional calculation for Understat stats
            # Create a temporary column that is NaN where data is missing
            temp_stat_col = f'temp_{col}'
            df[temp_stat_col] = np.where(df['understat_missing'] == 1, np.nan, df[col])

            # Calculate the rolling mean, which will automatically ignore NaNs
            # This correctly adjusts the denominator
            rolling_avg = df.groupby(['season_x', 'element'])[temp_stat_col].rolling(window=5, min_periods=1).mean()

            # The result of .rolling() has a multi-index, so we reset it to align with the main DataFrame
            df[form_col_name] = rolling_avg.reset_index(level=[0, 1], drop=True)

            # Clean up the temporary column
            df.drop(columns=[temp_stat_col], inplace=True)

            print(f"  > Created '{form_col_name}' column (conditional rolling average).")

        else:
            # Standard rolling average for non-Understat columns
            rolling_avg = df.groupby(['season_x', 'element'])[col].rolling(window=5, min_periods=1).mean()
            df[form_col_name] = rolling_avg.reset_index(level=[0, 1], drop=True)
            print(f"  > Created '{form_col_name}' column (standard rolling average).")

    # Fill any potential NaNs created by the rolling operation with 0
    form_cols = [f'form_{col}' for col in verified_stats]
    df[form_cols] = df[form_cols].fillna(0)

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