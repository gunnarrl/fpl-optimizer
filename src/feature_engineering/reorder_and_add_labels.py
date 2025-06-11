import pandas as pd
import numpy as np


def process_fpl_data(input_filename='your_data.csv', output_filename='processed_fpl_data.csv'):
    """
    Combines six operations:
    1. Filters out specified older seasons.
    2. Creates 'has_double_GW_next_5' flag.
    3. Creates target labels 'next_GW_points' and 'next_GW_minutes'.
    4. Filters out all data from Gameweek 38.
    5. Reorders all columns into a specific, final order.
    6. Saves the processed data to a new CSV file.
    """
    try:
        # Load the dataset
        df = pd.read_csv(input_filename)
        print(f"Successfully loaded '{input_filename}' with {len(df)} rows.")

        # --- PART 1: Filter Out Old Seasons ---
        seasons_to_remove = ['2016-17', '2017-18']
        df = df[~df['season_x'].isin(seasons_to_remove)]
        print(f"Removed seasons {seasons_to_remove}. Remaining rows: {len(df)}.")

        # --- PART 2: Create 'has_double_GW_next_5' Column ---
        schedule_df = df.groupby(['season_x', 'name', 'GW']).size().reset_index(name='n_matches')
        schedule_df['is_dgw'] = schedule_df['n_matches'] > 1
        groups = schedule_df.groupby(['season_x', 'name'])['is_dgw']
        dgw_t0 = schedule_df['is_dgw']
        dgw_t1 = groups.shift(-1).fillna(False)
        dgw_t2 = groups.shift(-2).fillna(False)
        dgw_t3 = groups.shift(-3).fillna(False)
        dgw_t4 = groups.shift(-4).fillna(False)
        schedule_df['has_double_GW_next_5'] = (dgw_t0 | dgw_t1 | dgw_t2 | dgw_t3 | dgw_t4).astype(int)
        df = pd.merge(df, schedule_df[['season_x', 'name', 'GW', 'has_double_GW_next_5']],
                      on=['season_x', 'name', 'GW'], how='left')
        print("Successfully created the 'has_double_GW_next_5' column.")

        # --- PART 3: Create the Target Label Columns ---
        df.sort_values(by=['season_x', 'name', 'GW'], inplace=True)
        # Create next_GW_points
        df['next_GW_points'] = df.groupby(['season_x', 'name'])['total_points'].shift(-1)
        print("Successfully created the 'next_GW_points' column.")
        # Create next_GW_minutes
        df['next_GW_minutes'] = df.groupby(['season_x', 'name'])['minutes'].shift(-1)
        print("Successfully created the 'next_GW_minutes' column.")


        # --- PART 4: Filter Out Gameweek 38 ---
        # This is done AFTER creating next gameweek columns so that GW 37 has a valid label.
        original_rows = len(df)
        df = df[df['GW'] != 38].copy()
        print(f"Removed all GW 38 data. Rows removed: {original_rows - len(df)}.")

        # --- PART 5: Reorder All Columns ---
        categorical_columns = [
            'team', 'GW', 'was_home', 'round', 'opponent_team', 'opp_team_name',
            'kickoff_time', 'element', 'season_x', 'name', 'position', 'team_x',
            'understat_missing', 'fixture'
        ]
        all_columns = df.columns.tolist()
        # Add new columns to the front of the list
        final_ordered_list = ['next_GW_points', 'next_GW_minutes', 'has_double_GW_next_5']

        for col in categorical_columns:
            if col in all_columns and col not in final_ordered_list:
                final_ordered_list.append(col)

        for col in all_columns:
            if col not in final_ordered_list:
                final_ordered_list.append(col)

        df = df[final_ordered_list]
        print("Successfully reordered all columns.")

        # --- PART 6: Save the Result ---
        df.to_csv(output_filename, index=False)
        print(f"Success! Final data saved to '{output_filename}'.")

    except FileNotFoundError:
        print(f"ERROR: The file '{input_filename}' was not found.")
    except KeyError as e:
        print(f"ERROR: A required column was not found: {e}. Please check your CSV. You may be missing a 'minutes' column.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- HOW TO USE THIS SCRIPT ---
# 1. Change 'your_data.csv' to the name of your file below.
# 2. Run the script.
process_fpl_data(input_filename='../../data/processed/master_with_team_data.csv')