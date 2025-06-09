import pandas as pd
import numpy as np
import os


def add_team_strength_and_form(input_filepath, team_list_filepath, output_filepath):
    """
    Adds team strength and form columns to the player data.
    This version uses kickoff_time to uniquely identify matches, correctly
    handling double gameweeks and providing more accurate stats.

    Args:
        input_filepath (str): Path to the master CSV file (e.g., master_with_form.csv).
        team_list_filepath (str): Path to the CSV file with team name to ID mappings.
        output_filepath (str): Path to save the final CSV with team strength columns.
    """
    # --- 1. Load the Datasets ---
    try:
        player_df = pd.read_csv(input_filepath)
        print(f"Successfully loaded player dataset from '{input_filepath}'. Initial shape: {player_df.shape}")
        # **MODIFICATION**: Ensure kickoff_time is a datetime object for proper sorting
        if 'kickoff_time' not in player_df.columns:
            raise ValueError("Input file must contain a 'kickoff_time' column.")
        player_df['kickoff_time'] = pd.to_datetime(player_df['kickoff_time'])

        team_list_df = pd.read_csv(team_list_filepath)
        print(f"Successfully loaded team list from '{team_list_filepath}'.")
    except FileNotFoundError as e:
        print(f"Error: A required file was not found. {e}")
        return
    except Exception as e:
        print(f"An error occurred while loading the files: {e}")
        return

    # --- 2. Map Team Names to Integer IDs ---
    print("\n--- Mapping team names to integer IDs ---")
    team_list_df = team_list_df[['season', 'team', 'team_name']]
    team_list_df.rename(columns={'season': 'season_x', 'team_name': 'team_x'}, inplace=True)
    player_df = pd.merge(player_df, team_list_df, on=['season_x', 'team_x'], how='left')
    print("  > 'team' column (integer ID) added to the player dataset.")

    # --- 3. Calculate Match Points ---
    conditions = [
        (player_df['was_home'] & (player_df['team_h_score'] > player_df['team_a_score'])),
        (~player_df['was_home'] & (player_df['team_a_score'] > player_df['team_h_score'])),
        (player_df['team_h_score'] == player_df['team_a_score'])
    ]
    choices = [3, 3, 1]
    player_df['match_points'] = np.select(conditions, choices, default=0)
    print("  > 'match_points' column created.")

    # --- 4. Create and build the Team-Match DataFrame (Robust Method) ---
    print("\n--- Creating and calculating all team-level statistics (per-match) ---")

    # **MODIFICATION**: Group by the unique match identifier (team and kickoff_time)
    # This prevents errors from double gameweeks. We keep 'GW' for later ranking context.
    team_match_df = player_df.groupby(['season_x', 'team', 'kickoff_time', 'GW']).agg(
        opponent_team=('opponent_team', 'first'),
        match_points=('match_points', 'first')
    ).reset_index()

    # Sort chronologically to correctly calculate cumulative and rolling stats
    team_match_df.sort_values(by=['season_x', 'team', 'kickoff_time'], inplace=True)
    print("  > Created team-level dataframe, unique per match.")

    # Calculate cumulative points and form based on match history
    team_match_df['cumulative_points'] = team_match_df.groupby(['season_x', 'team'])['match_points'].cumsum()
    team_match_df['team_form'] = team_match_df.groupby(['season_x', 'team'])['match_points'].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    print("  > Calculated 'cumulative_points' and 'team_form'.")

    # **MODIFICATION**: Robustly calculate team_strength for each gameweek
    # First, find the final cumulative points for each team in a gameweek (handles DGWs)
    final_gw_stats = team_match_df.sort_values('kickoff_time').groupby(['season_x', 'team', 'GW']).last().reset_index()
    # Then, rank teams within the gameweek based on that final point total
    final_gw_stats['team_strength'] = final_gw_stats.groupby(['season_x', 'GW'])['cumulative_points'].rank(method='min',
                                                                                                           ascending=False)

    # Merge the correct team_strength back to the per-match data
    team_match_df = pd.merge(
        team_match_df,
        final_gw_stats[['season_x', 'team', 'GW', 'team_strength']],
        on=['season_x', 'team', 'GW'],
        how='left'
    )
    print("  > Calculated 'team_strength' correctly for double gameweeks.")

    # Calculate opp_team_strength using the new strength lookup table
    strength_lookup_df = final_gw_stats[['season_x', 'GW', 'team', 'team_strength']].rename(
        columns={'team': 'opponent_team', 'team_strength': 'opp_team_strength'}
    )
    team_match_df = pd.merge(team_match_df, strength_lookup_df, on=['season_x', 'GW', 'opponent_team'], how='left')
    print("  > Calculated 'opp_team_strength'.")

    # Calculate upcoming_strength (this logic remains similar but acts on per-match data)
    upcoming_strengths_df = pd.DataFrame(index=team_match_df.index)
    for i in range(1, 6):
        upcoming_strengths_df[f'next_opp_strength_{i}'] = team_match_df.groupby(['season_x', 'team'])[
            'opp_team_strength'].shift(-i)
    team_match_df['upcoming_strength'] = upcoming_strengths_df.mean(axis=1)
    team_match_df['upcoming_strength'] = team_match_df.groupby(['season_x', 'team'])['upcoming_strength'].fillna(
        method='ffill')
    print("  > Calculated 'upcoming_strength'.")

    # --- 5. Perform a Single, Final Merge ---
    print("\n--- Merging all team-level stats back to the player dataset ---")

    # **MODIFICATION**: Merge on the unique match identifier
    cols_to_merge = [
        'season_x', 'kickoff_time', 'team',
        'team_strength', 'team_form', 'opp_team_strength', 'upcoming_strength'
    ]
    # Drop intermediate columns before merge to avoid conflicts
    team_match_df.drop(columns=['GW', 'match_points', 'cumulative_points', 'opponent_team'], inplace=True)

    final_df = pd.merge(player_df, team_match_df, on=['season_x', 'kickoff_time', 'team'], how='left')

    # --- 6. Validation Step ---
    print("\n--- Validating final output ---")
    if len(final_df) > len(player_df):
        print(
            f"CRITICAL WARNING: The number of rows increased from {len(player_df)} to {len(final_df)}. Duplicates were created.")
    else:
        print(f"  > Row count validation passed. Final row count is {len(final_df)}.")

    # --- 7. Save the Final Dataset ---
    try:
        output_dir = os.path.dirname(output_filepath)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        final_df.drop(columns=['match_points'], inplace=True)
        final_df.to_csv(output_filepath, index=False)
        print(f"\nSuccessfully saved the final dataset to '{output_filepath}'.")
        print(f"Final dataset shape: {final_df.shape}")
    except Exception as e:
        print(f"\nError: Could not save the final file. Reason: {e}")


if __name__ == '__main__':
    # Define the input and output filepaths
    input_file = '../../data/processed/master_with_form.csv'
    team_list_file = '../../data/raw/cleaned/master_team_list.csv'
    output_file_with_team_data = '../../data/processed/master_with_team_data.csv'

    # Run the function
    add_team_strength_and_form(input_file, team_list_file, output_file_with_team_data)