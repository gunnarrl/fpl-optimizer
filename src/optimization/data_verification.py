import os
import json
import pandas as pd

# Same paths as your main script
DATA_DIR = os.path.abspath(os.path.join(__file__, "..", "..", "..", "data"))
PREDICTIONS_FILE = os.path.join(DATA_DIR, "output", 'predictions', "koa_predictions_updated.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "output", "teams")
GW_START = 1

print("=== DATA VERIFICATION ===")

# Load CSV data
print("Loading CSV data...")
df = pd.read_csv(PREDICTIONS_FILE)
print(f"CSV shape: {df.shape}")
print(f"CSV columns: {list(df.columns)}")

# Check first few rows
print("\nFirst 3 rows of CSV:")
print(df.head(3))

# Get unique elements from CSV
csv_elements = set(df['element'].unique())
print(f"\nUnique elements in CSV: {len(csv_elements)}")
print(f"Sample element IDs from CSV: {sorted(list(csv_elements))[:10]}")

# Load initial squad JSON
initial_file = os.path.join(OUTPUT_DIR, f"team_gw{GW_START}.json")
print(f"\nLoading initial squad from: {initial_file}")

try:
    with open(initial_file, 'r') as f:
        initial_data = json.load(f)

    print(f"JSON structure keys: {list(initial_data.keys())}")
    print(f"Squad size in JSON: {len(initial_data['squad'])}")

    # Extract element IDs from JSON
    json_elements = {p['element'] for p in initial_data['squad']}
    print(f"Element IDs in JSON: {sorted(json_elements)}")

    # Check if JSON elements exist in CSV
    missing_in_csv = json_elements - csv_elements
    if missing_in_csv:
        print(f"ERROR: Elements in JSON but not in CSV: {missing_in_csv}")
    else:
        print("✓ All JSON elements found in CSV")

    # Check squad composition from JSON
    positions = {}
    teams = {}
    total_value = 0

    for player in initial_data['squad']:
        element_id = player['element']
        # Find this player in CSV
        player_data = df[df['element'] == element_id]
        if len(player_data) > 0:
            pos = player_data.iloc[0]['position']
            team = player_data.iloc[0]['team']
            value = player_data.iloc[0]['value']

            positions[pos] = positions.get(pos, 0) + 1
            teams[team] = teams.get(team, 0) + 1
            total_value += value

    print(f"\nSquad composition by position: {positions}")
    print(f"Squad composition by team: {teams}")
    print(f"Total squad value: {total_value}")

    # Check FPL rules
    print(f"\nFPL Rules Check:")
    print(f"GKP: {positions.get('GK', 0)} (need 2) - {'✓' if positions.get('GK', 0) == 2 else '✗'}")
    print(f"DEF: {positions.get('DEF', 0)} (need 5) - {'✓' if positions.get('DEF', 0) == 5 else '✗'}")
    print(f"MID: {positions.get('MID', 0)} (need 5) - {'✓' if positions.get('MID', 0) == 5 else '✗'}")
    print(f"FWD: {positions.get('FWD', 0)} (need 3) - {'✓' if positions.get('FWD', 0) == 3 else '✗'}")
    print(f"Total players: {sum(positions.values())} (need 15) - {'✓' if sum(positions.values()) == 15 else '✗'}")
    print(f"Budget: {total_value} (limit 1000) - {'✓' if total_value <= 1000 else '✗'}")

    team_violations = {t: count for t, count in teams.items() if count > 3}
    if team_violations:
        print(f"Team limit violations (>3 players): {team_violations}")
    else:
        print("✓ No team limit violations")

except Exception as e:
    print(f"Error: {e}")