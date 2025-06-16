import os
import json
import pandas as pd
import pulp

# Your existing configuration
TEAM = [
    'Matz Sels', 'Joe Gauci', 'Joško Gvardiol', 'Milos Kerkez',
    'Lino da Cruz Sousa', 'Daniel Muñoz', 'Nikola Milenković', 'Morgan Rogers',
    'Bryan Mbeumo', 'Mohamed Salah', 'Cole Palmer', 'Antoine Semenyo',
    'Erling Haaland', 'Chris Wood', 'Yoane Wissa'
]

DATA_DIR = os.path.abspath(os.path.join(__file__, "..", "..", "..", "data"))
PREDICTIONS_FILE = os.path.join(DATA_DIR, "output", 'predictions', "koa_predictions_updated.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "output", "teams")
PLANNING_HORIZON = 6
GW_START = 1
BUDGET = 1000
INITIAL_FREE_TRANSFERS = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load and process data
print("Loading data...")
df = pd.read_csv(PREDICTIONS_FILE)

# IMPORTANT FIX: Get unique player data only
# The issue is that your CSV has multiple rows per player (likely one per gameweek)
# We need to get unique players only for the constraints
print("Processing unique players...")
unique_players_df = df.drop_duplicates(subset=['element'])
print(f"Total rows in CSV: {len(df)}")
print(f"Unique players: {len(unique_players_df)}")

# Use unique players for building the optimization model
players = unique_players_df['element'].unique().tolist()
teams = unique_players_df['team'].unique().tolist()
GWs = list(range(GW_START, GW_START + PLANNING_HORIZON))

# Create dictionaries using unique player data
cost = unique_players_df.set_index('element')['value'].to_dict()
pos = unique_players_df.set_index('element')['position'].to_dict()
team_name = unique_players_df.set_index('element')['team'].to_dict()

print(f"Players: {len(players)}")
print(f"Teams: {len(teams)}")
print(f"Position distribution: {unique_players_df['position'].value_counts().to_dict()}")

# Load initial squad - FIXED VERSION
initial_file = os.path.join(OUTPUT_DIR, f"team_gw{GW_START}.json")
try:
    with open(initial_file, 'r') as f:
        initial_data = json.load(f)

    # CORRECT: Extract element IDs from the squad list
    initial_squad = {p['element'] for p in initial_data['squad']}
    print(f"Loaded initial squad from {initial_file}")
    print(f"Initial squad element IDs: {sorted(initial_squad)}")
    print(f"Initial squad size: {len(initial_squad)}")

    # VERIFY: Check the initial squad against unique players
    initial_squad_players = unique_players_df[unique_players_df['element'].isin(initial_squad)]
    print(f"Initial squad verification:")
    print(f"  Position counts: {initial_squad_players['position'].value_counts().to_dict()}")
    print(f"  Team distribution: {initial_squad_players['team'].value_counts().to_dict()}")
    print(f"  Total value: {initial_squad_players['value'].sum()}")

except FileNotFoundError:
    print(f"Warning: Initial squad file {initial_file} not found.")
    initial_squad = set()
except Exception as e:
    print(f"Error loading initial squad: {e}")
    initial_squad = set()

# Transform predicted points - FIXED VERSION
print("Transforming predicted points...")
pred = {}
act = {}

for gw in GWs:
    pred_col = f'predicted_points_gw+{gw}'
    act_col = f'points_gw+{gw}'

    if pred_col in df.columns:
        # Group by element and take the first non-null value for each player
        pred_gw = df.groupby('element')[pred_col].first().fillna(0).to_dict()
        for player in players:
            pred[(gw, player)] = pred_gw.get(player, 0)
    else:
        print(f"Warning: Column {pred_col} not found")
        for player in players:
            pred[(gw, player)] = 0

    if act_col in df.columns:
        act_gw = df.groupby('element')[act_col].first().fillna(0).to_dict()
        for player in players:
            act[(gw, player)] = act_gw.get(player, 0)
    else:
        print(f"Warning: Column {act_col} not found")
        for player in players:
            act[(gw, player)] = 0

# Now build the optimization model
print("Building optimization model...")
model = pulp.LpProblem(name="fpl_optimizer", sense=pulp.LpMaximize)

# Variables
in_team = pulp.LpVariable.dicts("in_team", (GWs, players), cat="Binary")
starting = pulp.LpVariable.dicts("starting", (GWs, players), cat="Binary")
captain = pulp.LpVariable.dicts("captain", (GWs, players), cat="Binary")
vice = pulp.LpVariable.dicts("vice", (GWs, players), cat="Binary")
buy = pulp.LpVariable.dicts("buy", (GWs, players), cat="Binary")
sell = pulp.LpVariable.dicts("sell", (GWs, players), cat="Binary")
transfers_made = pulp.LpVariable.dicts("transfers_made", GWs, lowBound=0, cat="Integer")
free_transfers = pulp.LpVariable.dicts("free_transfers", GWs, lowBound=0, upBound=2, cat="Integer")
points_hit = pulp.LpVariable.dicts("points_hit", GWs, lowBound=0, cat="Integer")

# Objective
model += (
        pulp.lpSum(starting[g][p] * pred.get((g, p), 0) + captain[g][p] * pred.get((g, p), 0)
                   for g in GWs for p in players)
        - pulp.lpSum(points_hit[g] * 4 for g in GWs)
)

# Constraints
for g in GWs:
    # Squad constraints
    model += pulp.lpSum(in_team[g][p] for p in players) == 15
    model += pulp.lpSum(starting[g][p] for p in players) == 11
    model += pulp.lpSum(in_team[g][p] * cost[p] for p in players) <= BUDGET

    # Position constraints - FIXED: using correct position names
    # Check what position names are actually in your data
    position_names = unique_players_df['position'].unique()
    print(f"Available positions: {position_names}")

    # Adjust position names based on your data
    if 'GK' in position_names:  # Your data uses 'GK' not 'GKP'
        model += pulp.lpSum(in_team[g][p] for p in players if pos[p] == 'GK') == 2
        model += pulp.lpSum(starting[g][p] for p in players if pos[p] == 'GK') == 1
    elif 'GKP' in position_names:
        model += pulp.lpSum(in_team[g][p] for p in players if pos[p] == 'GKP') == 2
        model += pulp.lpSum(starting[g][p] for p in players if pos[p] == 'GKP') == 1

    model += pulp.lpSum(in_team[g][p] for p in players if pos[p] == 'DEF') == 5
    model += pulp.lpSum(in_team[g][p] for p in players if pos[p] == 'MID') == 5
    model += pulp.lpSum(in_team[g][p] for p in players if pos[p] == 'FWD') == 3

    # Formation constraints
    model += pulp.lpSum(starting[g][p] for p in players if pos[p] == 'DEF') >= 3
    model += pulp.lpSum(starting[g][p] for p in players if pos[p] == 'MID') >= 2
    model += pulp.lpSum(starting[g][p] for p in players if pos[p] == 'FWD') >= 1

    # Team constraints
    for t in teams:
        model += pulp.lpSum(in_team[g][p] for p in players if team_name[p] == t) <= 3

    # Captaincy constraints
    model += pulp.lpSum(captain[g][p] for p in players) == 1
    model += pulp.lpSum(vice[g][p] for p in players) == 1
    for p in players:
        model += captain[g][p] + vice[g][p] <= starting[g][p]
        model += starting[g][p] <= in_team[g][p]

    # Transfer logic - SIMPLIFIED for debugging
    if g == GW_START:
        # Fix initial squad
        if initial_squad:
            for p in players:
                if p in initial_squad:
                    model += in_team[g][p] == 1
                else:
                    model += in_team[g][p] == 0

        # No transfers in first gameweek
        model += transfers_made[g] == 0
        model += free_transfers[g] == INITIAL_FREE_TRANSFERS
        model += points_hit[g] == 0

        # Ensure buy/sell are 0 for first gameweek
        for p in players:
            model += buy[g][p] == 0
            model += sell[g][p] == 0
    else:
        # Simplified transfer logic for subsequent gameweeks
        for p in players:
            model += in_team[g][p] == in_team[g - 1][p] + buy[g][p] - sell[g][p]
            model += buy[g][p] + sell[g][p] <= 1  # Can't buy and sell same player

        # Transfer counting - FIXED: using integer arithmetic
        model += 2 * transfers_made[g] == pulp.lpSum(buy[g][p] + sell[g][p] for p in players)

        # Simplified free transfer logic
        model += free_transfers[g] == 1  # Always 1 free transfer for simplicity

        # Points hit
        model += points_hit[g] >= transfers_made[g] - free_transfers[g]

# Solve
print("Solving the optimization problem...")
model.solve(pulp.PULP_CBC_CMD(msg=True))
print(f"Status: {pulp.LpStatus[model.status]}")

if pulp.LpStatus[model.status] == 'Optimal':
    print("Success! Saving results...")
    # Your existing result saving code here
    for g in GWs:
        squad = []
        total_pred = 0
        total_actual = 0
        for p in players:
            if in_team[g][p].value() and in_team[g][p].value() > 0.5:
                player_name = unique_players_df[unique_players_df['element'] == p]['name'].iloc[0]
                player_pos = pos[p]
                player_team = team_name[p]

                gw_pred = pred.get((g, p), 0)
                gw_act = act.get((g, p), 0)

                info = {
                    'element': p,
                    'name': player_name,
                    'position': player_pos,
                    'team': player_team,
                    'was_starting': bool(starting[g][p].value() and starting[g][p].value() > 0.5),
                    'is_captain': bool(captain[g][p].value() and captain[g][p].value() > 0.5),
                    'is_vice': bool(vice[g][p].value() and vice[g][p].value() > 0.5),
                    'predicted': gw_pred,
                    'actual': gw_act
                }
                if info['was_starting']:
                    mult = 2 if info['is_captain'] else 1
                    total_pred += info['predicted'] * mult
                    total_actual += info['actual'] * mult
                squad.append(info)

        output = {
            'GW': g,
            'free_transfers': int(free_transfers[g].value()) if free_transfers[g].value() else 0,
            'transfers_made': int(transfers_made[g].value()) if transfers_made[g].value() else 0,
            'points_hit': int(points_hit[g].value() * 4) if points_hit[g].value() else 0,
            'total_predicted_points': total_pred,
            'total_actual_points': total_actual,
            'squad': sorted(squad, key=lambda x: (x['position'], x['name']))
        }

        file_path = os.path.join(OUTPUT_DIR, f"team_gw{g}.json")
        print(f"Saving optimal squad for GW{g} to {file_path}")
        with open(file_path, 'w') as f:
            json.dump(output, f, indent=2)
else:
    print("Could not find an optimal solution.")
    print(f"Status: {pulp.LpStatus[model.status]}")