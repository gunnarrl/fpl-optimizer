import pandas as pd
import numpy as np


def process_and_merge_data(fpl_filepath, understat_filepath, playerlist_filepath, output_filepath,
                           target_season_context='2024-25'):
    """
    Merges historical FPL and Understat data for players listed for a target season context,
    handles missing Understat data based on minutes played, and saves the result.

    Args:
        fpl_filepath (str): Filepath for the cleaned FPL data CSV (historical data).
        understat_filepath (str): Filepath for the Understat data CSV (historical data).
        playerlist_filepath (str): Filepath for the player ID list CSV (e.g., for 2024-25 players).
        output_filepath (str): Filepath for the output master CSV file.
        target_season_context (str): The season context for which players are being selected (e.g., '2024-25').
                                     This is used for identifying relevant players, not for filtering historical data by a future season.
    """
    try:
        # Load the datasets
        fpl_df = pd.read_csv(fpl_filepath)
        understat_df = pd.read_csv(understat_filepath)
        player_list_df = pd.read_csv(playerlist_filepath)
        print("Successfully loaded all CSV files.")
    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure the file paths are correct.")
        return
    except Exception as e:
        print(f"An error occurred during file loading: {e}")
        return

    # --- 1. Prepare Player List for Filtering ---
    try:
        player_list_df['first_name'] = player_list_df['first_name'].astype(str).fillna('')
        player_list_df['second_name'] = player_list_df['second_name'].astype(str).fillna('')
        player_list_df['full_name'] = player_list_df['first_name'] + " " + player_list_df['second_name']
        player_list_df['full_name'] = player_list_df['full_name'].str.strip()
        current_season_player_names = set(player_list_df['full_name'])
        if not current_season_player_names:
            print(f"Warning: The player list from '{playerlist_filepath}' is empty.")
        print(
            f"Loaded {len(current_season_player_names)} unique player names for whom to gather historical data (context: {target_season_context}).")
    except KeyError as e:
        print(f"Error: Missing expected columns ('first_name', 'second_name') in '{playerlist_filepath}'. {e}")
        return
    except Exception as e:
        print(f"An error occurred during player list preparation: {e}")
        return

    # --- 2. Filter FPL Data & Create Merge Key ---
    original_fpl_columns = list(fpl_df.columns)
    fpl_df_filtered_by_player = fpl_df[fpl_df['name'].isin(current_season_player_names)].copy()

    # *** FIX START: Create a mergeable date column from FPL 'kickoff_time' ***
    try:
        fpl_df_filtered_by_player['merge_date'] = pd.to_datetime(
            fpl_df_filtered_by_player['kickoff_time']).dt.date.astype(str)
    except KeyError:
        print("Error: 'kickoff_time' column not found in FPL data. Cannot create date-based merge key.")
        return
    # *** FIX END ***

    if fpl_df_filtered_by_player.empty:
        print(
            f"No historical FPL data found for any of the {len(current_season_player_names)} players in the player list. Output file will likely be empty.")
        pd.DataFrame(columns=original_fpl_columns + ['xG', 'xA', 'key_passes', 'npg', 'npxG', 'xGChain', 'xGBuildup',
                                                     'understat_missing']).to_csv(output_filepath, index=False)
        return
    print(f"Found {fpl_df_filtered_by_player.shape[0]} historical FPL data rows for players in the player list.")
    fpl_df_to_merge = fpl_df_filtered_by_player

    # --- 3. Prepare Understat Data & Create Merge Key ---
    TEAM_NAME_MAPPINGS = {
        "Manchester City": "Man City", "Manchester United": "Man Utd", "Newcastle United": "Newcastle",
        "Nottingham Forest": "Nott'm Forest", "Sheffield United": "Sheffield Utd", "Tottenham": "Spurs",
        "West Bromwich Albion": "West Brom", "Wolverhampton Wanderers": "Wolves"
    }
    understat_df['h_team'] = understat_df['h_team'].replace(TEAM_NAME_MAPPINGS)
    understat_df['a_team'] = understat_df['a_team'].replace(TEAM_NAME_MAPPINGS)

    # *** FIX START: Create a mergeable date column from Understat 'date' and prepare for merge ***
    try:
        understat_df['merge_date'] = pd.to_datetime(understat_df['date']).dt.date.astype(str)
    except KeyError:
        print("Error: 'date' column not found in Understat data. Cannot create date-based merge key.")
        return

    understat_df.rename(columns={'fpl_id': 'element'}, inplace=True)
    understat_cols_to_merge = [
        'element', 'merge_date', 'xG', 'xA',
        'key_passes', 'npg', 'npxG', 'xGChain', 'xGBuildup',
        'h_team', 'a_team'
    ]
    # *** FIX END ***

    understat_cols_to_merge = [col for col in understat_cols_to_merge if col in understat_df.columns]
    understat_subset = understat_df[understat_cols_to_merge]

    # --- 4. Merge DataFrames ---
    # *** FIX: Changed the merge key from ['season_x', 'GW', 'element'] to ['element', 'merge_date'] ***
    master_df = pd.merge(fpl_df_to_merge, understat_subset, on=['element', 'merge_date'], how='left')
    print("Historical FPL and Understat data merged successfully using player ID and date.")

    # --- 5. Fill Missing 'team_x' Values ---
    if 'h_team' in master_df.columns and 'a_team' in master_df.columns:
        master_df['team_x'] = np.where(
            master_df['team_x'].isnull(),
            np.where(master_df['was_home'], master_df['h_team'], master_df['a_team']),
            master_df['team_x']
        )
        print("Filled missing 'team_x' values using merged Understat team names.")
    else:
        print("Warning: 'h_team' or 'a_team' not found in merged data. Skipping 'team_x' fill for merged rows.")

    # --- 6. Handle Missing Understat Data ---
    understat_value_columns = ['xG', 'xA', 'key_passes', 'npg', 'npxG', 'xGChain', 'xGBuildup']

    for col in understat_value_columns:
        if col not in master_df.columns:
            master_df[col] = np.nan

    master_df['played_but_missing_understat'] = (
            (master_df['minutes'] > 0) & (master_df['xG'].isnull())
    ).astype(int)

    # --- Step 3: Impute values based on playing time ---

    # Condition 1: Player did not play (minutes == 0).
    # The correct value for their performance stats is 0.
    did_not_play_mask = (master_df['minutes'] == 0)
    master_df.loc[did_not_play_mask, understat_value_columns] = \
        master_df.loc[did_not_play_mask, understat_value_columns].fillna(0)

    # Condition 2: Player PLAYED but data is missing.
    # This is a true missing value. We will leave it as NaN for now
    # so that models like XGBoost/LightGBM can handle it optimally.
    # This section is intentionally left blank as we handle the final fill in the next step.

    # --- Step 4: Final Imputation for any remaining NaNs ---
    # This now only affects the 'played_but_missing_understat' cases.
    # Instead of a blanket '0' or '-1', we can make a more intelligent choice.

    # OPTION A (Recommended for XGBoost/LightGBM): Leave as NaN
    # No code needed here if you are using these models. They will handle it.

    # OPTION C (Advanced): Impute with a rolling player average
    # This is the most complex but potentially most accurate method.
    # It assumes the player performed at their recent average.
    # for col in understat_value_columns:
    #     # Group by player, calculate rolling mean over 5 games, then backfill NaNs
    #     rolling_mean = master_df.groupby('name')[col].transform(lambda x: x.rolling(5, min_periods=1).mean())
    #     master_df[col] = master_df[col].fillna(rolling_mean)
    #     # Fill any remaining NaNs for players with no history with 0
    #     master_df[col] = master_df[col].fillna(0)

    # --- 7. Define Final Columns and Save ---
    # We add the new understat columns, but no need to add 'merge_date' to the final output.
    output_columns = original_fpl_columns[:]
    for col in understat_value_columns:
        if col not in output_columns:
            output_columns.append(col)
    if 'understat_missing' not in output_columns:
        output_columns.append('understat_missing')

    for col in output_columns:
        if col not in master_df:
            master_df[col] = 0

    master_final_df = master_df[output_columns]
    master_final_df.to_csv(output_filepath, index=False)
    print(f"Successfully created the master combined file at: {output_filepath}")
    print(f"Final dataset dimensions: {master_final_df.shape}")


if __name__ == '__main__':
    player_list_file = '../../data/raw/2024-25/player_idlist.csv'
    fpl_data_file = '../../data/raw/cleaned/cleaned_merged_seasons_team_aggregated.csv'
    understat_data_file = '../../data/raw/cleaned/understat_2020_24.csv'
    output_file = '../../data/raw/cleaned/master_combined.csv'
    current_fpl_season = '2024-25'

    process_and_merge_data(fpl_data_file, understat_data_file, player_list_file, output_file,
                           target_season_context=current_fpl_season)
