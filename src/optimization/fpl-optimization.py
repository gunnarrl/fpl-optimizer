import os
import json
import pandas as pd
import pulp

# Fixed syntax error: missing comma
TEAM = [
    'Matz Sels', 'Karl Hein', 'Virgil van Dijk', 'Milos Kerkez',
    'Lino da Cruz Sousa', 'Gabriel Magalhães', 'Trent Alexander-Arnold', 'Jarrod Bowen',
    'Bryan Mbeumo', 'Mohamed Salah', 'Cole Palmer', 'Antoine Semenyo',
    'Erling Haaland', 'Chris Wood', 'Yoane Wissa'
]

# CONFIGURATION
DATA_DIR = os.path.abspath(os.path.join(__file__, "..", "..", "..", "data"))
# ASSUMPTION: The predictions file has columns like 'predicted_points_gw+1', 'predicted_points_gw+2', etc.
# and 'points_gw+1', 'points_gw+2', etc.
PREDICTIONS_FILE = os.path.join(DATA_DIR, "output", 'predictions', "koa_predictions_updated.csv")
OUTPUT_DIR = os.path.join(DATA_DIR, "output", "teams")
PLANNING_HORIZON = 6  # number of GWs ahead to optimize
GW_START = 1  # starting gameweek
BUDGET = 1000  # example budget x 10
INITIAL_FREE_TRANSFERS = 1  # Number of free transfers at GW_START

# UTILITIES
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- DATA LOADING ---
# READ AND PREPROCESS DATA
print("Loading data...")
df = pd.read_csv(PREDICTIONS_FILE)

# Transform the wide format data to long format
print("Transforming data from wide to long format...")
players = df['element'].unique().tolist()
teams = df['team'].unique().tolist()
GWs = list(range(GW_START, GW_START + PLANNING_HORIZON))

# Create dictionaries for player attributes (cost, position, team) - these are static
cost = df.set_index('element')['value'].to_dict()
pos = df.set_index('element')['position'].to_dict()
team_name = df.set_index('element')['team'].to_dict()

# Transform predicted points and actual points from wide to long format
pred = {}
act = {}

for gw in GWs:
    pred_col = f'predicted_points_gw+{gw}'
    act_col = f'points_gw+{gw}'

    # Check if columns exist before processing
    if pred_col in df.columns:
        for _, row in df.iterrows():
            pred[(gw, row['element'])] = row[pred_col] if pd.notnull(row[pred_col]) else 0
    else:
        print(f"Warning: Column {pred_col} not found in data")
        for player in players:
            pred[(gw, player)] = 0

    if act_col in df.columns:
        for _, row in df.iterrows():
            act[(gw, row['element'])] = row[act_col] if pd.notnull(row[act_col]) else 0
    else:
        print(f"Warning: Column {act_col} not found in data")
        for player in players:
            act[(gw, player)] = 0

# LOAD PRESELECTED GW1 SQUAD
initial_file = os.path.join(OUTPUT_DIR, f"team_gw{GW_START}.json")
try:
    with open(initial_file, 'r') as f:
        initial_data = json.load(f)
    initial_squad = {p['element'] for p in initial_data['squad']}
    print(f"Loaded initial squad from {initial_file}")
except FileNotFoundError:
    print(f"Warning: Initial squad file {initial_file} not found. Creating empty initial squad.")
    initial_squad = set()
except Exception as e:
    print(f"Error loading initial squad: {e}")
    initial_squad = set()

# --- BUILD MODEL ---
print("Building optimization model...")
model = pulp.LpProblem(name="fpl_optimizer", sense=pulp.LpMaximize)

# VARIABLES
in_team = pulp.LpVariable.dicts("in_team", (GWs, players), cat="Binary")
starting = pulp.LpVariable.dicts("starting", (GWs, players), cat="Binary")
captain = pulp.LpVariable.dicts("captain", (GWs, players), cat="Binary")
vice = pulp.LpVariable.dicts("vice", (GWs, players), cat="Binary")
buy = pulp.LpVariable.dicts("buy", (GWs, players), cat="Binary")
sell = pulp.LpVariable.dicts("sell", (GWs, players), cat="Binary")
transfers_made = pulp.LpVariable.dicts("transfers_made", GWs, lowBound=0, cat="Integer")
free_transfers = pulp.LpVariable.dicts("free_transfers", GWs, lowBound=0, upBound=2, cat="Integer")
points_hit = pulp.LpVariable.dicts("points_hit", GWs, lowBound=0, cat="Integer")

# OBJECTIVE: maximize predicted points, minus transfer hits
model += (
        pulp.lpSum(starting[g][p] * pred.get((g, p), 0) + captain[g][p] * pred.get((g, p), 0)
                   for g in GWs for p in players)
        - pulp.lpSum(points_hit[g] * 4 for g in GWs)  # 4 points per hit
)

# --- CONSTRAINTS ---
for g in GWs:
    # --- CORE SQUAD AND TEAM CONSTRAINTS ---
    # Squad size must be 15
    model += pulp.lpSum(in_team[g][p] for p in players) == 15
    # Starting XI size must be 11
    model += pulp.lpSum(starting[g][p] for p in players) == 11
    # Budget constraint for each gameweek
    model += pulp.lpSum(in_team[g][p] * cost[p] for p in players) <= BUDGET

    # --- POSITIONAL & FORMATION CONSTRAINTS ---
    # Squad composition rules (2 GK, 5 DEF, 5 MID, 3 FWD)
    model += pulp.lpSum(in_team[g][p] for p in players if pos[p] == 'GKP') == 2
    model += pulp.lpSum(in_team[g][p] for p in players if pos[p] == 'DEF') == 5
    model += pulp.lpSum(in_team[g][p] for p in players if pos[p] == 'MID') == 5
    model += pulp.lpSum(in_team[g][p] for p in players if pos[p] == 'FWD') == 3

    # Starting XI formation rules (1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD)
    model += pulp.lpSum(starting[g][p] for p in players if pos[p] == 'GKP') == 1
    model += pulp.lpSum(starting[g][p] for p in players if pos[p] == 'DEF') >= 3
    model += pulp.lpSum(starting[g][p] for p in players if pos[p] == 'MID') >= 2
    model += pulp.lpSum(starting[g][p] for p in players if pos[p] == 'FWD') >= 1

    # --- TEAM LIMIT CONSTRAINT ---
    # Max 3 players from any single team
    for t in teams:
        model += pulp.lpSum(in_team[g][p] for p in players if team_name[p] == t) <= 3

    # --- CAPTAINCY CONSTRAINTS ---
    model += pulp.lpSum(captain[g][p] for p in players) == 1
    model += pulp.lpSum(vice[g][p] for p in players) == 1
    for p in players:
        model += captain[g][p] + vice[g][p] <= starting[g][p]
        model += starting[g][p] <= in_team[g][p]

    # --- TRANSFER LOGIC ---
    model += transfers_made[g] == 0.5 * pulp.lpSum(buy[g][p] + sell[g][p] for p in players)

    if g == GW_START:
        # Handle initial squad setup
        if initial_squad:
            # Fix the initial squad for the first gameweek
            for p in players:
                if p in initial_squad:
                    model += in_team[g][p] == 1
                else:
                    model += in_team[g][p] == 0
        else:
            print("Warning: No initial squad provided, optimizer will select starting squad")

        # Set initial free transfers and no transfers made in GW_START
        model += free_transfers[g] == INITIAL_FREE_TRANSFERS
        model += transfers_made[g] == 0
    else:
        # Link squad composition between gameweeks via transfers
        for p in players:
            model += in_team[g][p] == in_team[g - 1][p] + buy[g][p] - sell[g][p]
            # A player can either be bought or sold in a gameweek, not both
            model += buy[g][p] + sell[g][p] <= 1

        # Free transfer logic: min(2, previous_ft - transfers_made + 1)
        # This is modeled with two constraints. The variable's upBound of 2 handles the ceiling.
        model += free_transfers[g] <= free_transfers[g - 1] - transfers_made[g - 1] + 1
        model += free_transfers[g] >= 1  # You always get at least 1 new free transfer

    # Points hit logic: hits are incurred for transfers exceeding the free allowance
    # Using an intermediate variable to model this robustly
    transfers_over_limit = pulp.LpVariable(f"transfers_over_limit_gw{g}", lowBound=0, cat="Integer")
    model += transfers_made[g] - free_transfers[g] <= transfers_over_limit
    model += points_hit[g] == transfers_over_limit

# --- SOLVE AND SAVE RESULTS ---
print("Solving the optimization problem...")
model.solve(pulp.PULP_CBC_CMD(msg=True))
print(f"Status: {pulp.LpStatus[model.status]}")

# EXTRACT AND SAVE RESULTS
if pulp.LpStatus[model.status] == 'Optimal':
    for g in GWs:
        squad = []
        total_pred = 0
        total_actual = 0
        for p in players:
            if in_team[g][p].value() and in_team[g][p].value() > 0.5:
                # Get the player's static info
                player_name = df[df['element'] == p]['name'].iloc[0]
                player_pos = pos[p]
                player_team = team_name[p]

                # Get the player's points for the specific gameweek
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
            'points_hit': int(points_hit[g].value() * 4) if points_hit[g].value() else 0,  # show total points hit
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
    print("This could be due to:")
    print("- Infeasible constraints (e.g., budget too low, no valid initial squad)")
    print("- Missing data in the CSV file")
    print("- Incorrect column names or data format")