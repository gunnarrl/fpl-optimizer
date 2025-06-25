import pandas as pd
import json
import os

# --- USER INPUT ---
# The list of 15 players for the initial squad.
# Note: Names must match the 'name' column in your CSV.
# Common FPL names like 'Gabriel dos Santos Magalhães' are often just 'Gabriel'.
# I have adjusted this below, but you may need to check your CSV for others.
PLAYER_NAMES = [
    'Matz Sels', 'Joe Gauci', 'Joško Gvardiol', 'Milos Kerkez',
    'Lino da Cruz Sousa', 'Daniel Muñoz', 'Nikola Milenković', 'Morgan Rogers',
    'Bryan Mbeumo', 'Mohamed Salah', 'Cole Palmer', 'Antoine Semenyo',
    'Erling Haaland', 'Chris Wood', 'Yoane Wissa'
]

# Define your captain and vice-captain
CAPTAIN = 'Mohamed Salah'
VICE_CAPTAIN = 'Erling Haaland'
BUDGET = 1000.0

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
    Generates a JSON file for an initial FPL squad for GW1 based on the new format.
    """
    try:
        # Load the player data from the CSV file
        df = pd.read_csv("../../data/processed/processed_fpl_data.csv")
    except FileNotFoundError:
        print("Error: 'processed_fpl_data.csv' not found.")
        print("Please ensure the script is in the correct directory relative to the data file.")
        return

    # Optional: Run the query to show the lowest value players
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
    total_cost = squad_df['value'].sum()
    bank = BUDGET - total_cost

    # Define the columns for the planning horizon (next 6 GWs)
    future_gw_cols = [f'points_gw+{i}' for i in range(1, 7)]

    # Initialize totals
    total_predicted_points_current_gw = 0
    total_actual_points_current_gw = 0
    total_predicted_points_planning_horizon = 0

    # --- JSON CONSTRUCTION ---
    squad_list_for_json = []
    starting_xi_elements = []

    for _, player in squad_df.iterrows():
        is_starter = player['name'] in starting_11
        is_captain = player['name'] == captain
        is_vice_captain = player['name'] == vice_captain

        # Calculate predicted points over the planning horizon for each player
        player_planning_total = sum(player.get(col, 0) for col in future_gw_cols)

        # Create the player info dictionary with the new format
        player_info = {
            'element': int(player['element']),
            'name': player['name'],
            'position': player['position'],
            'team': player['team_x'],
            'cost': player.get('value', 0),
            'predicted_points_current_gw': player.get('xP'),
            'actual_points_current_gw': player.get('total_points', 0),
            'predicted_points_planning_horizon': player_planning_total,
            'is_starting': is_starter,
            'is_captain': is_captain,
            'is_vice': is_vice_captain
        }
        squad_list_for_json.append(player_info)

        # Add to totals and starting XI list if the player is a starter
        if is_starter:
            starting_xi_elements.append(int(player['element']))
            multiplier = 2 if is_captain else 1

            total_predicted_points_current_gw += player_info['predicted_points_current_gw'] * multiplier
            total_actual_points_current_gw += player_info['actual_points_current_gw'] * multiplier
            total_predicted_points_planning_horizon += player_info['predicted_points_planning_horizon'] * multiplier

    # Assemble the final JSON object in the requested format
    output_json = {
        'GW': 1,
        'transfers_made': 0,
        'points_hit': 0,
        'bank': bank,
        'chips_used': None,  # No chip used in GW1
        'transfers_in': {},   # No transfers in for the initial squad
        'transfers_out': {},  # No transfers out for the initial squad
        'total_predicted_points_current_gw': total_predicted_points_current_gw,
        'total_actual_points_current_gw': total_actual_points_current_gw,
        'total_predicted_points_planning_horizon': total_predicted_points_planning_horizon,
        'squad': sorted(squad_list_for_json, key=lambda x: (x['position'], x['name'])),
        'starting_xi': starting_xi_elements
    }

    # Save the JSON file
    file_path = "../../data/output/teams/xgb/team_gw1.json"
    print(f"\nSaving initial squad for GW1 to {file_path}")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(output_json, f, indent=2)

    # --- PRINT SUMMARY ---
    print("\n--- Initial Squad Summary ---")
    print(f"Total Team Cost: £{total_cost:.1f}m")
    print(f"Remaining Bank: £{bank:.1f}m")
    print(f"Total Predicted Points (GW1): {total_predicted_points_current_gw:.2f}")
    print(f"Total Predicted Points (Planning Horizon GWs 2-7): {total_predicted_points_planning_horizon:.2f}")
    print("-----------------------------\n")


if __name__ == '__main__':
    # Determine the starting 11 based on who is NOT on the bench
    bench = ['Joe Gauci', 'Nikola Milenković', 'Lino da Cruz Sousa', 'Morgan Rogers']
    starting_11 = [p for p in PLAYER_NAMES if p not in bench]

    create_initial_squad_json(PLAYER_NAMES, starting_11, CAPTAIN, VICE_CAPTAIN)