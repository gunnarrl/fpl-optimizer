import pandas as pd
import json
import os

# --- USER INPUT ---
# The list of 15 players for the initial squad.
# Note: Names must match the 'name' column in your CSV.
# Common FPL names like 'Gabriel dos Santos Magalhães' are often just 'Gabriel'.
# I have adjusted this below, but you may need to check your CSV for others.
PLAYER_NAMES = [
    'Matz Sels', 'Karl Hein', 'Virgil van Dijk', 'Milos Kerkez',
    'Lino da Cruz Sousa', 'Gabriel Magalhães', 'Trent Alexander-Arnold', 'Jarrod Bowen',
    'Bryan Mbeumo', 'Mohamed Salah', 'Cole Palmer', 'Antoine Semenyo',
    'Erling Haaland', 'Chris Wood', 'Yoane Wissa'
]

# Define your starting 11, captain, and vice-captain
STARTING_11 = [
    'Matz Sels', 'Virgil van Dijk', 'Gabriel Magalhães', 'Trent Alexander-Arnold',
    'Bukayo Saka', 'Bryan Mbeumo', 'Mohamed Salah', 'Cole Palmer',
    'Antoine Semenyo', 'Erling Haaland', 'Chris Wood'
]

CAPTAIN = 'Mohamed Salah'
VICE_CAPTAIN = 'Erling Haaland'


# --- SCRIPT LOGIC ---

def print_lowest_value_players(df):
    """
    Finds and prints the 4 lowest value players for each position for GW1.
    """
    try:
        # Filter for the correct gameweek and season
        gw1_df = df[(df['GW'] == 1) & (df['season_x'] == '2024-25')].copy()

        if gw1_df.empty:
            print("\nCould not find any GW1 2024-25 data to find lowest value players.")
            return

        # Get the 4 players with the smallest value for each position
        lowest_value_players = gw1_df.groupby('position').apply(
            lambda x: x.nsmallest(4, 'value')
        ).reset_index(drop=True)

        print("\n--- 4 Lowest Value Players by Position (GW1 2024-25) ---")
        # Sort by position for a consistent output order
        for position in sorted(lowest_value_players['position'].unique()):
            players = lowest_value_players[lowest_value_players['position'] == position]
            print(f"\n{position}:")
            for _, player in players.iterrows():
                print(f"  - {player['name']} ({player['team_x']}): £{player['value']:.1f}m")
        print("--------------------------------------------------------\n")

    except Exception as e:
        print(f"\nCould not generate lowest value players list due to an error: {e}")


def create_initial_squad_json(player_names, starting_11, captain, vice_captain):
    """
    Generates a JSON file for an initial FPL squad for GW1.
    """
    try:
        # Load the player data from the CSV file
        df = pd.read_csv("../../data/processed/processed_fpl_data.csv")
    except FileNotFoundError:
        print("Error: 'processed_fpl_data.csv' not found.")
        print("Please ensure the script is in the same directory as the data file.")
        return

    # Run the query to show the lowest value players
    print_lowest_value_players(df)

    # Filter for Gameweek 1 data for the selected players for the 2024-25 season
    squad_df = df[(df['GW'] == 1) & (df['name'].isin(player_names)) & (df['season_x'] == '2024-25')].copy()

    # --- VALIDATION ---
    if len(squad_df) != 15:
        print(f"Warning: Found {len(squad_df)} out of 15 players for GW1 in the CSV.")
        found_players = squad_df['name'].tolist()
        missing_players = [p for p in player_names if p not in found_players]
        if missing_players:
            print(f"Missing players: {missing_players}")
        print("Please check the names in the PLAYER_NAMES list against your CSV.")
        if len(squad_df) == 0:
            return

    # --- CALCULATIONS ---

    # 1. Calculate total team cost for GW1
    total_cost = squad_df['value'].sum()

    # 2. Calculate total future predicted points (GW+1 to GW+6)
    future_gw_cols = [f'points_gw+{i}' for i in range(1, 7)]
    total_future_predicted_points = 0

    # 3. Calculate total predicted and actual points for the GW1 JSON file
    total_gw1_predicted_points = 0
    total_gw1_actual_points = 0

    # --- JSON CONSTRUCTION ---
    squad_list_for_json = []

    for _, player in squad_df.iterrows():
        is_starter = player['name'] in starting_11
        is_captain = player['name'] == captain
        is_vice_captain = player['name'] == vice_captain

        info = {
            'element': int(player['element']),
            'name': player['name'],
            'position': player['position'],
            'team': player['team_x'],  # Assuming 'team_x' is the correct team name column
            'value': player.get('value', 0),
            'was_starting': is_starter,
            'is_captain': is_captain,
            'is_vice': is_captain,
            'predicted': player.get('xP', 0),  # Use xP for single GW prediction
            'actual': player.get('total_points', 0)  # Actual points for GW1
        }
        squad_list_for_json.append(info)

        # Add to totals if the player is in the starting XI
        if is_starter:
            multiplier = 2 if is_captain else 1

            # Sum up future points for the separate calculation
            player_future_points = sum(player.get(col, 0) for col in future_gw_cols)
            total_future_predicted_points += player_future_points * multiplier

            # Sum up GW1 predicted and actual points for the JSON output
            total_gw1_predicted_points += info['predicted'] * multiplier
            total_gw1_actual_points += info['actual'] * multiplier


    # Assemble the final JSON object
    output_json = {
        'GW': 1,
        'free_transfers': 1,  # Default for start of season
        'transfers_made': 0,
        'points_hit': 0,
        'total_predicted_points': total_gw1_predicted_points,
        'total_actual_points': total_gw1_actual_points,
        'total_value': total_cost,
        'squad': sorted(squad_list_for_json, key=lambda x: (x['position'], x['name']))
    }

    # Save the JSON file
    file_path = "../../data/output/teams/team_gw1.json"
    print(f"\nSaving initial squad for GW1 to {file_path}")
    with open(file_path, 'w') as f:
        json.dump(output_json, f, indent=2)

    # --- PRINT SUMMARY ---
    print("\n--- Initial Squad Summary ---")
    print(f"Total Team Cost: £{total_cost:.1f}m")
    print(f"Total Predicted Points (for GWs 2-7): {total_future_predicted_points:.2f}")
    print("-----------------------------\n")


if __name__ == '__main__':
    # Determine the starting 11 based on who is NOT on the bench
    bench = ['Karl Hein', 'Milos Kerkez', 'Lino da Cruz Sousa', 'Jarrod Bowen']
    starting_11 = [p for p in PLAYER_NAMES if p not in bench]

    create_initial_squad_json(PLAYER_NAMES, starting_11, CAPTAIN, VICE_CAPTAIN)