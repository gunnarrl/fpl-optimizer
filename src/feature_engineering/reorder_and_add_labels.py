import pandas as pd
import numpy as np

# --- 1. Define Column Categories based on your full list ---
PLANNING_HORIZON = 6

def _create_dgw_row(group: pd.DataFrame) -> pd.Series:
    """
    Helper function to process a single player's DGW, combining match stats
    into a single representative row.
    """
    # Ensure matches are sorted by time to correctly identify the first and second game
    group = group.sort_values('kickoff_time', ascending=True)
    first_match = group.iloc[0]
    second_match = group.iloc[-1] # Use -1 to handle rare triple gameweeks

    # A dictionary to build the new, combined row
    new_row = {}

    # == Define Column Lists based on Aggregation Rules ==

    # 1. Stats to be summed across all matches in the gameweek
    sum_cols = [
        'assists', 'bonus', 'bps', 'clean_sheets', 'creativity', 'goals_conceded',
        'goals_scored', 'ict_index', 'influence', 'minutes', 'own_goals',
        'penalties_missed', 'penalties_saved', 'red_cards', 'saves',
        'team_a_score', 'team_h_score', 'threat', 'total_points', 'yellow_cards',
        'xG', 'xA', 'key_passes', 'npg', 'npxG', 'xGChain', 'xGBuildup', 'xP'
    ]

    # 2. Stats to take from the first match
    first_match_cols = [
        'season_x', 'name', 'position', 'team_x', 'element', 'fixture', 'kickoff_time',
        'transfers_balance', 'transfers_in', 'transfers_out', 'value', 'was_home',
        'round', 'GW', 'team', 'team_form', 'team_strength', 'upcoming_strength'
    ]

    # 3. Stats to take from the second (or last) match
    second_match_cols = [
        'points', 'team_goals_scored', 'team_goals_conceded', 'team_goals_diff',
        'aggregate_xG', 'aggregate_xA', 'aggregate_key_passes', 'aggregate_npg',
        'aggregate_npxG', 'aggregate_xGChain', 'aggregate_xGBuildup', 'aggregate_xP',
        'aggregate_goals_scored', 'aggregate_saves', 'aggregate_penalties_saved',
        'aggregate_assists', 'aggregate_total_points', 'aggregate_minutes',
        'aggregate_own_goals', 'aggregate_penalties_missed', 'aggregate_clean_sheets',
        'aggregate_threat', 'aggregate_ict_index', 'aggregate_influence',
        'aggregate_creativity', 'per_90_xG', 'per_90_xA', 'per_90_key_passes',
        'per_90_npg', 'per_90_npxG', 'per_90_xGChain', 'per_90_xGBuildup', 'per_90_xP',
        'per_90_goals_scored', 'per_90_saves', 'per_90_penalties_saved',
        'per_90_assists', 'per_90_total_points', 'per_90_own_goals',
        'per_90_penalties_missed', 'per_90_clean_sheets', 'per_90_threat',
        'per_90_ict_index', 'per_90_influence', 'per_90_creativity', 'form_3_xP',
        'form_3_goals_scored', 'form_3_saves', 'form_3_penalties_saved', 'form_3_assists',
        'form_3_xG', 'form_3_xA', 'form_3_total_points', 'form_3_minutes',
        'form_3_key_passes', 'form_3_npg', 'form_3_npxG', 'form_3_xGChain',
        'form_3_xGBuildup', 'form_3_own_goals', 'form_3_penalties_missed',
        'form_3_clean_sheets', 'form_3_bps', 'form_3_threat', 'form_3_ict_index',
        'form_3_influence', 'form_3_creativity', 'form_5_xP', 'form_5_goals_scored',
        'form_5_saves', 'form_5_penalties_saved', 'form_5_assists', 'form_5_xG',
        'form_5_xA', 'form_5_total_points', 'form_5_minutes', 'form_5_key_passes',
        'form_5_npg', 'form_5_npxG', 'form_5_xGChain', 'form_5_xGBuildup',
        'form_5_own_goals', 'form_5_penalties_missed', 'form_5_clean_sheets',
        'form_5_bps', 'form_5_threat', 'form_5_ict_index', 'form_5_influence',
        'form_5_creativity'
    ]
    # Handle 'understat_missing' column if it exists, as it was not specified.
    # A reasonable assumption is to take the value from the last match.
    if 'understat_missing' in group.columns:
        second_match_cols.append('understat_missing')

    # == Apply Aggregation Rules ==
    for col in sum_cols:
        new_row[col] = group[col].sum()
    for col in first_match_cols:
        new_row[col] = first_match[col]
    for col in second_match_cols:
        new_row[col] = second_match[col]

    # Handle opponent-specific stats for each match
    new_row['opponent_team_1'] = first_match['opponent_team']
    new_row['opp_team_name_1'] = first_match['opp_team_name']
    new_row['opp_team_strength_1'] = first_match['opp_team_strength']
    new_row['opponent_team_2'] = second_match['opponent_team']
    new_row['opp_team_name_2'] = second_match['opp_team_name']
    new_row['opp_team_strength_2'] = second_match['opp_team_strength']

    # Add the DGW flag
    new_row['is_dgw'] = 1

    return pd.Series(new_row)

def combine_dgw(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes a DataFrame of FPL stats to combine rows for double gameweeks (DGWs).

    Args:
        df: A pandas DataFrame with player-match data.

    Returns:
        A pandas DataFrame where DGWs are consolidated into a single row,
        and an 'is_dgw' column indicates the row type.
    """
    df = df.copy()

    # Identify DGWs by finding duplicates for a player ('element') in a given gameweek ('GW')
    dgw_identifiers = ['season_x', 'element', 'GW']
    dgw_mask = df.duplicated(subset=dgw_identifiers, keep=False)

    # Separate single gameweek (SGW) and DGW data
    sgw_df = df[~dgw_mask].copy()
    dgw_df = df[dgw_mask].copy()

    # Process SGW rows to match the final column structure
    sgw_df['is_dgw'] = 0
    sgw_df.rename(columns={
        'opponent_team': 'opponent_team_1',
        'opp_team_name': 'opp_team_name_1',
        'opp_team_strength': 'opp_team_strength_1'
    }, inplace=True)
    # Add null columns for the second match opponent, which doesn't exist for SGWs
    sgw_df['opponent_team_2'] = np.nan
    sgw_df['opp_team_name_2'] = np.nan
    sgw_df['opp_team_strength_2'] = np.nan

    # Process DGW rows if any exist
    if not dgw_df.empty:
        # Group by the DGW identifiers and apply the helper function to each group
        processed_dgw_df = dgw_df.groupby(dgw_identifiers, as_index=False).apply(_create_dgw_row)

        # Combine the processed DGWs with the SGWs
        final_df = pd.concat([sgw_df, processed_dgw_df], ignore_index=True, sort=False)
    else:
        # If no DGWs were found, the result is just the formatted SGW dataframe
        final_df = sgw_df

    return final_df


def process_fpl_data(input_filename='your_data.csv', output_filename='processed_fpl_data.csv'):
    """
    Processes FPL data:
    1. Consolidates Double Gameweeks.
    2. Filters out specified older seasons.
    3. Creates specific feature flags for future Double Gameweeks.
    4. Creates target labels for a multi-week planning horizon.
    5. Filters out data from the final gameweek.
    6. Reorders all columns into a specific, final order.
    7. Saves the processed data to a new CSV file.
    """
    try:
        df = pd.read_csv(input_filename)
        print(f"Successfully loaded '{input_filename}' with {len(df)} rows.")

        df = combine_dgw(df)

        seasons_to_remove = ['2016-17', '2017-18']
        df = df[~df['season_x'].isin(seasons_to_remove)]
        print(f"Removed seasons {seasons_to_remove}. Remaining rows: {len(df)}.")

        # --- FIX STARTS HERE ---

        # Sort values to ensure chronological order before any processing
        df.sort_values(['season_x', 'name', 'GW'], inplace=True)

        # Create a complete timeline for each player to handle missing gameweeks
        def create_complete_timeline(group):
            season = group['season_x'].iloc[0]
            name = group['name'].iloc[0]
            min_gw = group['GW'].min()
            max_gw = group['GW'].max()

            # Create a full gameweek range for the player
            full_gw_range = pd.RangeIndex(start=min_gw, stop=max_gw + 1, name='GW')
            group = group.set_index('GW').reindex(full_gw_range)

            # Forward-fill essential identifiers
            group['name'].fillna(name, inplace=True)
            group['season_x'].fillna(season, inplace=True)
            # Fill 'is_dgw' for missing weeks with 0, as they are SGWs by definition
            group['is_dgw'].fillna(0, inplace=True)
            return group.reset_index()

        # Apply the timeline correction
        df = df.groupby(['season_x', 'element'], group_keys=False).apply(create_complete_timeline)
        print("Created complete timelines for all players to ensure accurate shifts.")

        # Now, create future DGW flags and target labels
        grouped = df.groupby(['season_x', 'element'])

        # 1. Create future DGW flags
        new_dgw_flag_data = {}
        dgw_flag_cols = []
        is_dgw_series = grouped['is_dgw']
        for i in range(1, PLANNING_HORIZON + 1):
            col_name = f'is_dgw_in_gw+{i}'
            new_dgw_flag_data[col_name] = is_dgw_series.shift(-i).fillna(0).astype(int)
            dgw_flag_cols.append(col_name)

        dgw_flags_df = pd.DataFrame(new_dgw_flag_data, index=df.index)
        df = pd.concat([df, dgw_flags_df], axis=1)
        print(f"Successfully created {PLANNING_HORIZON} future Double Gameweek flags.")

        # 2. Create target labels (points and minutes)
        points = grouped['total_points']
        minutes = grouped['minutes']

        new_labels_data = {}
        new_label_cols = []
        for i in range(1, PLANNING_HORIZON + 1):
            points_col_name = f'points_gw+{i}'
            minutes_col_name = f'minutes_gw+{i}'

            new_labels_data[points_col_name] = points.shift(-i)
            new_labels_data[minutes_col_name] = minutes.shift(-i)
            new_label_cols.extend([points_col_name, minutes_col_name])

        labels_df = pd.DataFrame(new_labels_data, index=df.index)
        df = pd.concat([df, labels_df], axis=1)
        print(f"Successfully created target labels for a planning horizon of {PLANNING_HORIZON} weeks.")

        # Remove the placeholder rows created by reindexing
        df.dropna(subset=['fixture'],
                  inplace=True)  # 'fixture' is a reliable column that is NaN only on placeholder rows
        print("Removed placeholder rows after creating labels.")

        # --- FIX ENDS HERE ---

        # Nullify labels that extend beyond the end of the season
        for i in range(1, PLANNING_HORIZON + 1):
            points_col_name = f'points_gw+{i}'
            minutes_col_name = f'minutes_gw+{i}'
            invalid_horizon_mask = (df['GW'] + i) > 38
            df.loc[invalid_horizon_mask, [points_col_name, minutes_col_name]] = np.nan

        # ... (rest of your script remains the same)
        categorical_columns = [
            'team', 'GW', 'was_home', 'round', 'opponent_team_1', 'opponent_team_2',
            'kickoff_time', 'element', 'season_x', 'name', 'position', 'team_x',
            'understat_missing', 'fixture'
        ]
        # Rename opponent columns to match the list
        df.rename(columns={
            'opponent_team_1': 'opponent_1_team',
            'opponent_team_2': 'opponent_2_team'
        }, inplace=True)

        all_columns = df.columns.tolist()
        final_ordered_list = new_label_cols + dgw_flag_cols + ['is_dgw']

        for col in categorical_columns:
            if col in all_columns and col not in final_ordered_list:
                final_ordered_list.append(col)

        for col in all_columns:
            if col not in final_ordered_list:
                final_ordered_list.append(col)

        df = df[final_ordered_list]
        print("Successfully reordered all columns.")

        df.to_csv(output_filename, index=False)
        print(f"Success! Final data saved to '{output_filename}'.")

    except FileNotFoundError:
        print(f"ERROR: The file '{input_filename}' was not found.")
    except KeyError as e:
        print(f"ERROR: A required column was not found: {e}. Please check your CSV.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == '__main__':
    process_fpl_data(input_filename='../../data/processed/master_with_team_data.csv',
                     output_filename='../../data/processed/processed_fpl_data.csv')