import pandas as pd
import pulp

TEAM = ['Sels', 'Sanchez' 'Virgil', 'Kerkez', 'Tarkowski', 'Gabriel', 'Trent', 'Bukayo Saka', 'Bruno Fernandes', 'Salah', 'Palmer', 'Semenyo', 'Haaland', 'Wood', 'Isak' ]
def optimize_fpl_squad(predictions_df, target_gw):
    """
    Optimizes an FPL squad based on player predictions for a specific gameweek.

    Args:
        predictions_df (pd.DataFrame): DataFrame with player predictions.
        target_gw (int): The gameweek to optimize for.

    Returns:
        tuple: A tuple containing the starting XI, bench, and a summary of the team.
    """

    # Task 1: Filter the prediction DataFrame to a single gameweek
    gw_df = predictions_df[predictions_df['GW'] == target_gw].copy()

    # If there are no players for the given gameweek, return empty results
    if gw_df.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    # Task 2: Keep only relevant columns, including next_GW_points
    player_pool = gw_df[['element', 'name', 'position', 'value', 'team', 'predicted_points', 'next_GW_points']].copy()

    # Task 3: Value is already in the correct integer form (e.g., 40 for 4.0M)
    player_pool.rename(columns={'value': 'cost'}, inplace=True)

    # Task 4: Define player position categories
    position_map = {'GK': 1, 'DEF': 2, 'MID': 3, 'FWD': 4}
    player_pool['position_id'] = player_pool['position'].map(position_map)

    # Task 5: Define position and squad constraints
    BUDGET = 1000
    SQUAD_SIZE = 15
    POSITION_CONSTRAINTS = {1: 2, 2: 5, 3: 5, 4: 3}
    MAX_PLAYERS_PER_TEAM = 3

    # Initialize the optimization problem
    prob = pulp.LpProblem(f"FPL_Squad_Optimization_GW_{target_gw}", pulp.LpMaximize)

    # Task 6: Set up binary decision variables for each player
    player_vars = pulp.LpVariable.dicts("Player", player_pool['element'], 0, 1, pulp.LpBinary)

    # Task 7: Define the objective function (maximize current GW predicted points)
    prob += pulp.lpSum([player_vars[p['element']] * p['predicted_points'] for _, p in player_pool.iterrows()])

    # Task 8: Add total cost constraint
    prob += pulp.lpSum(
        [player_vars[p['element']] * p['cost'] for _, p in player_pool.iterrows()]) <= BUDGET, "TotalCost"

    # Task 9: Add squad size constraint
    prob += pulp.lpSum([player_vars[p['element']] for _, p in player_pool.iterrows()]) == SQUAD_SIZE, "SquadSize"

    # Task 10: Add positional constraints
    for pos_id, count in POSITION_CONSTRAINTS.items():
        prob += pulp.lpSum([player_vars[p['element']] for _, p in player_pool.iterrows() if
                            p['position_id'] == pos_id]) == count, f"Position_{pos_id}"

    # Task 11: Add max 3 players per real-life team constraint
    for team_id in player_pool['team'].unique():
        prob += pulp.lpSum([player_vars[p['element']] for _, p in player_pool.iterrows() if
                            p['team'] == team_id]) <= MAX_PLAYERS_PER_TEAM, f"Team_{team_id}"

    # Task 12: Solve the optimization problem
    prob.solve(pulp.PULP_CBC_CMD(msg=0))  # Suppress solver messages

    # Check if an optimal solution was found
    if prob.status != pulp.LpStatusOptimal:
        return pd.DataFrame(), pd.DataFrame(), {}

    # Task 13: Get the final squad
    selected_elements = [p['element'] for _, p in player_pool.iterrows() if player_vars[p['element']].varValue == 1]
    final_squad = player_pool[player_pool['element'].isin(selected_elements)]

    # Task 14: Calculate total predicted points, cost, and breakdown
    total_predicted_points = final_squad['predicted_points'].sum()
    total_cost = final_squad['cost'].sum()

    summary = {
        "Total Predicted Points": total_predicted_points,
        "Total Cost": total_cost / 10,
        "Position Breakdown": final_squad['position'].value_counts().to_dict()
    }

    # Task 15: Separate starting XI vs bench based on predicted_points
    final_squad = final_squad.sort_values(by='predicted_points', ascending=False)
    starting_xi = final_squad.head(11)
    bench = final_squad.tail(4)

    # Task 16: Add captain selection
    captain = starting_xi.iloc[0]
    vice_captain = starting_xi.iloc[1]

    # Adjust total points for captain
    summary["Total Predicted Points (with Captain)"] = total_predicted_points + captain['predicted_points']
    summary["Captain"] = captain['name']
    summary["Vice-Captain"] = vice_captain['name']

    return starting_xi, bench, summary


if __name__ == '__main__':
    try:
        # Load the predictions from the CSV file
        predictions_df = pd.read_csv('../../data/output/predictions/predictions_xgb.csv')

        # --- Handle Double Gameweeks ---
        # Group by player element and gameweek, summing points for DGWs.
        # Keep other essential info like name, position, etc.
        agg_dict = {
            'predicted_points': 'sum',
            'next_GW_points': 'sum',
            'name': 'first',
            'position': 'first',
            'value': 'first',
            'team': 'first'
        }
        processed_df = predictions_df.groupby(['element', 'GW']).agg(agg_dict).reset_index()

        # Initialize the total points tracker
        cumulative_total_points = 0.0

        # --- Loop through all gameweeks from 1 to 37 ---
        for target_gameweek in range(1, 38):
            print(f"\n{'=' * 20} Optimizing for Gameweek {target_gameweek} {'=' * 20}")

            # Run the optimization for the current gameweek
            starting_xi, bench, summary = optimize_fpl_squad(processed_df, target_gameweek)

            # Check if a valid squad was returned
            if not starting_xi.empty:
                # Display the optimization summary for the current gameweek
                print("\n--- Gameweek Squad Summary ---")
                for key, value in summary.items():
                    print(f"{key}: {value}")

                print("\n--- Starting XI ---")
                print(starting_xi[['name', 'position', 'team', 'predicted_points', 'next_GW_points', 'cost']])

                print("\n--- Bench ---")
                print(bench[['name', 'position', 'team', 'predicted_points', 'next_GW_points', 'cost']])

                # --- Calculate and add next_GW_points to the running total ---
                captain = starting_xi.iloc[0]
                # Sum the next_GW_points for the starting XI and double the captain's contribution
                gameweek_next_points = starting_xi['next_GW_points'].sum() + captain['next_GW_points']
                cumulative_total_points += gameweek_next_points

                print("\n--- Points Projection ---")
                print(
                    f"Projected points from Starting XI for next Gameweek (Captain Doubled): {gameweek_next_points:.2f}")
                print(f"CUMULATIVE TOTAL POINTS after Gameweek {target_gameweek}: {cumulative_total_points:.2f}")
            else:
                print(
                    f"\nCould not form a valid squad for Gameweek {target_gameweek}. It's possible there is insufficient player data for this week.")

        print(f"\n{'=' * 20} FINAL CUMULATIVE SCORE: {cumulative_total_points:.2f} {'=' * 20}")

    except FileNotFoundError:
        print("Error: 'predictions.csv' not found. Please ensure the file is in the correct directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")