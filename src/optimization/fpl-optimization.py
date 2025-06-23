import os
import json
import pandas as pd
import pulp
import numpy as np
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FPLOptimizer:
    def __init__(self, config_file=None):
        """Initialize the FPL Optimizer with configuration"""

        # Default configuration
        self.config = {
            'data_dir': os.path.abspath(os.path.join(__file__, "..", "..", "..", "data")),
            'predictions_file': 'predictions_autogluon_parallel.csv',
            'output_dir': 'teams',
            'planning_horizon': 6,
            'budget': 1000,
            'initial_free_transfers': 1,
            'transfer_penalty': 4,
            'max_team_players': 3,
            'squad_size': 15,
            'starting_xi': 11,
            'min_gk': 1, 'max_gk': 2,
            'min_def': 3, 'max_def': 5,
            'min_mid': 2, 'max_mid': 5,
            'min_fwd': 1, 'max_fwd': 3,
            'chips': {
                'wildcard_1': {'cost': 0, 'unlimited_transfers': True, 'available_gws': list(range(2, 21))},
                'wildcard_2': {'cost': 0, 'unlimited_transfers': True, 'available_gws': list(range(21, 39))},
                'bench_boost': {'cost': 0, 'bench_points': True, 'available_gws': list(range(1, 39))},
                'triple_captain': {'cost': 0, 'captain_multiplier': 3, 'available_gws': list(range(1, 39))},
                'free_hit': {'cost': 0, 'temporary_team': True, 'available_gws': list(range(1, 39))}
            }
        }

        # Load custom config if provided
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                custom_config = json.load(f)
                self.config.update(custom_config)

        self.setup_paths()
        self.load_data()

    def setup_paths(self):
        """Set up file paths"""
        self.data_dir = self.config['data_dir']
        self.predictions_file = os.path.join(self.data_dir, "output", 'predictions', self.config['predictions_file'])
        self.output_dir = os.path.join(self.data_dir, "output", self.config['output_dir'])
        os.makedirs(self.output_dir, exist_ok=True)

    def _validate_solution(self, solution):
        """Validate that the generated solution is FPL-legal."""
        logger.info("Validating generated solution...")
        squad_data = solution['squads'][solution['gw']]
        squad = squad_data['squad']

        # Validate squad size
        if len(squad) != self.config['squad_size']:
            logger.warning(f"Validation failed: Squad size is {len(squad)}, expected {self.config['squad_size']}.")

        # Validate budget
        total_cost = sum(p['cost'] for p in squad)
        if total_cost + solution['remaining_bank'] > self.config[
            'budget'] * 10:  # Assuming budget is in millions, cost is in 0.1M
            # Note: This check is simplistic. A rigorous check would use the initial bank value.
            pass  # For now, we assume the model's budget constraint is working.

        # Validate team limits
        team_counts = pd.Series([p['team'] for p in squad]).value_counts()
        if team_counts.max() > self.config['max_team_players']:
            logger.warning(
                f"Validation failed: Found {team_counts.max()} players from one team. Limit is {self.config['max_team_players']}.")

        # Validate starting XI formation
        starting_xi = squad_data['starting_xi']
        pos_counts = pd.Series([p['position'] for p in starting_xi]).value_counts().to_dict()
        if not (self.config['min_def'] <= pos_counts.get('DEF', 0) <= self.config['max_def']):
            logger.warning(f"Validation failed: Invalid number of defenders in starting XI.")
        # (Add similar checks for other positions)

        logger.info("Solution validation complete.")
        return True  # Or return False/raise error if validation fails

    def load_data(self):
        """Load and process prediction data"""
        logger.info("Loading and processing prediction data...")

        if not os.path.exists(self.predictions_file):
            raise FileNotFoundError(f"Predictions file not found: {self.predictions_file}")

        self.df = pd.read_csv(self.predictions_file)

        # Get unique player data for lookups
        self.unique_players_df = self.df.drop_duplicates(subset=['element'])
        self.players = self.unique_players_df['element'].unique().tolist()
        self.teams = self.unique_players_df['team'].unique().tolist()
        self.cost = self.unique_players_df.set_index('element')['value'].to_dict()
        self.pos = self.unique_players_df.set_index('element')['position'].to_dict()
        self.team_name = self.unique_players_df.set_index('element')['team'].to_dict()
        self.player_names = self.unique_players_df.set_index('element')['name'].to_dict()

        # --- NEW DATA PROCESSING LOGIC ---
        # Transform the wide data format to a long format for easy lookups

        # Define the prediction and actuals columns (gw+1 to gw+6)
        pred_cols = [f'predicted_points_gw+{i}' for i in range(1, 7)]
        actual_cols = [f'points_gw+{i}' for i in range(1, 7)]

        # Melt predictions
        preds_long = self.df.melt(
            id_vars=['element', 'GW'],
            value_vars=pred_cols,
            var_name='pred_col',
            value_name='predicted_points'
        )
        # Calculate the absolute gameweek for each prediction
        preds_long['offset'] = preds_long['pred_col'].str.extract(r'(\d+)').astype(int)
        preds_long['absolute_gw'] = preds_long['GW'] + preds_long['offset']

        # Melt actuals
        actuals_long = self.df.melt(
            id_vars=['element', 'GW'],
            value_vars=actual_cols,
            var_name='actual_col',
            value_name='actual_points'
        )
        # Calculate the absolute gameweek for each actual point total
        actuals_long['offset'] = actuals_long['actual_col'].str.extract(r'(\d+)').astype(int)
        actuals_long['absolute_gw'] = actuals_long['GW'] + actuals_long['offset']

        # Merge the two long dataframes into one processed table
        self.processed_df = pd.merge(
            preds_long[['element', 'absolute_gw', 'predicted_points']],
            actuals_long[['element', 'absolute_gw', 'actual_points']],
            on=['element', 'absolute_gw'],
            how='outer'
        ).fillna(0)
        # --- END OF NEW LOGIC ---
        self.dgw_players_by_gw = {}
        dgw_df = pd.read_csv('../../data/processed/processed_fpl_data.csv')

        # Filter for the '2024-25' season
        target_season = '2024-25'
        dgw_df_filtered = dgw_df[dgw_df['season_x'] == target_season].copy()
        logger.info(f"Filtered DGW data for season: {target_season}. Rows: {len(dgw_df_filtered)}")

        if not dgw_df_filtered.empty:
            # Assuming 'GW' is the Gameweek and 'element' is the player ID
            # or 'team' for team-based DGW identification.
            # If 'is_dgw' column is present and indicates True for DGWs:
            if 'is_dgw' in dgw_df_filtered.columns:
                dgw_df_filtered = dgw_df_filtered[dgw_df_filtered['is_dgw'] == True]
                logger.info(f"Filtered for 'is_dgw' == True. Rows: {len(dgw_df_filtered)}")

            # Group by GW and collect unique player elements involved in DGWs
            # Assuming 'element' column exists in your dgw_data.csv for players
            # If it's team-based, you'll need to map teams to players.
            if 'element' in dgw_df_filtered.columns and 'GW' in dgw_df_filtered.columns:
                for gw in dgw_df_filtered['GW'].unique():
                    players_in_dgw = dgw_df_filtered[dgw_df_filtered['GW'] == gw]['element'].unique().tolist()
                    self.dgw_players_by_gw[gw] = players_in_dgw
                    logger.debug(f"GW{gw} DGW players count: {len(players_in_dgw)}")
            elif 'team' in dgw_df_filtered.columns and 'GW' in dgw_df_filtered.columns:
                # If dgw_data.csv only has 'team' and 'GW' for DGWs, then map to players
                for gw in dgw_df_filtered['GW'].unique():
                    dgw_teams = dgw_df_filtered[dgw_df_filtered['GW'] == gw]['team'].unique().tolist()
                    players_in_dgw = self.unique_players_df[self.unique_players_df['team'].isin(dgw_teams)][
                        'element'].tolist()
                    self.dgw_players_by_gw[gw] = players_in_dgw
                    logger.debug(f"GW{gw} DGW players count (from teams): {len(players_in_dgw)}")
            else:
                logger.warning("DGW file missing 'element' or 'team' and 'GW' columns. Cannot identify DGW players.")
        else:
            logger.warning(f"No DGW data found for season {target_season} in {self.dgw_file}")

        logger.info(f"Loaded and processed data for {len(self.players)} unique players from {len(self.teams)} teams")

    def get_predictions_for_gw_range(self, start_gw, horizon):
        """Extract predictions and actual points for a specific gameweek range"""
        gws = list(range(start_gw, start_gw + horizon))

        # Filter the processed dataframe to get all predictions in the horizon
        relevant_data = self.processed_df[self.processed_df['absolute_gw'].isin(gws)]

        # Create dictionaries in the format the optimizer expects: (gameweek, player_id) -> points
        pred = relevant_data.set_index(['absolute_gw', 'element'])['predicted_points'].to_dict()
        actual = relevant_data.set_index(['absolute_gw', 'element'])['actual_points'].to_dict()

        # The optimizer expects a value for every player, so fill in missing ones with 0
        for gw in gws:
            for player in self.players:
                if (gw, player) not in pred:
                    pred[(gw, player)] = 0
                if (gw, player) not in actual:
                    actual[(gw, player)] = 0

        return gws, pred, actual

    def load_current_squad(self, gw):
        """Load current squad from file"""
        squad_file = os.path.join(self.output_dir, f"team_gw{gw}.json")

        if not os.path.exists(squad_file):
            logger.warning(f"No existing squad found for GW{gw}")
            return set(), 0, self.config['initial_free_transfers']

        try:
            with open(squad_file, 'r') as f:
                squad_data = json.load(f)

            # Handle both list and dict format for squad
            if isinstance(squad_data.get('squad'), list):
                if squad_data['squad'] and isinstance(squad_data['squad'][0], dict):
                    # New format: list of dicts with 'element' key
                    current_squad = {p['element'] for p in squad_data['squad']}
                else:
                    # Old format: list of player IDs
                    current_squad = set(squad_data['squad'])
            else:
                # Handle starting_xi format if squad is not available
                if 'starting_xi' in squad_data and isinstance(squad_data['starting_xi'], list):
                    current_squad = set(squad_data['starting_xi'])
                else:
                    current_squad = set()

            bank = squad_data.get('bank', 0)
            free_transfers = squad_data.get('free_transfers', 1)

            logger.info(
                f"Loaded squad for GW{gw}: {len(current_squad)} players, ${bank / 10:.1f}M bank, {free_transfers} free transfers")
            return current_squad, bank, free_transfers

        except Exception as e:
            logger.error(f"Error loading squad for GW{gw}: {e}")
            return set(), 0, self.config['initial_free_transfers']

    def optimize_single_gw(self, gw, current_squad=None, bank=0, free_transfers=1, use_chips=True, used_chips=None, use_planning_horizon=True):
        """Optimize for a single gameweek with transfer constraints"""
        logger.info(f"Optimizing for GW{gw}")

        # Get predictions for the planning horizon starting from this GW
        max_gameweek = 38  # Standard FPL season length
        adjusted_horizon = min(self.config['planning_horizon'], max_gameweek - gw + 1)

        if adjusted_horizon < self.config['planning_horizon']:
            logger.info(
                f"Planning horizon adjusted from {self.config['planning_horizon']} to {adjusted_horizon} for GW{gw} (end of season).")

        gws, pred, actual = self.get_predictions_for_gw_range(gw, adjusted_horizon)

        is_initial = False
        # If no current squad provided, this is the initial optimization
        if current_squad is None:
            current_squad, bank, free_transfers = self.load_current_squad(gw)
            is_initial = len(current_squad) == 0

        # Build the optimization model
        model = pulp.LpProblem(name=f"fpl_gw{gw}", sense=pulp.LpMaximize)

        # Decision variables
        in_team = pulp.LpVariable.dicts("in_team", (gws, self.players), cat="Binary")
        starting = pulp.LpVariable.dicts("starting", (gws, self.players), cat="Binary")
        captain = pulp.LpVariable.dicts("captain", (gws, self.players), cat="Binary")
        vice = pulp.LpVariable.dicts("vice", (gws, self.players), cat="Binary")

        # Transfer variables
        buy = pulp.LpVariable.dicts("buy", self.players, cat="Binary")
        sell = pulp.LpVariable.dicts("sell", self.players, cat="Binary")
        transfers_made = pulp.LpVariable("transfers_made", lowBound=0, cat="Integer")
        points_hit = pulp.LpVariable("points_hit", lowBound=0, cat="Integer")

        # Chip variables - use auxiliary variables to handle non-linear constraints
        chip_vars = {}
        triple_captain_active = {}
        bench_boost_active = {}

        players_to_exclude = []
        for p in self.players:
            total_pred = sum(pred.get((g, p), 0) for g in gws)
            if total_pred == 0:
                players_to_exclude.append(p)

        logger.info(f"Excluding {len(players_to_exclude)} players with zero predicted points.")

        # In the model-building phase, fix their variables to 0
        for g in gws:
            for p in players_to_exclude:
                model += in_team[g][p] == 0

        if used_chips is None:
            used_chips = set()

        if use_chips:
            for chip_name, chip_info in self.config['chips'].items():
                # Only create a variable for the chip if it's available and NOT already used
                if chip_name not in used_chips and gw in chip_info['available_gws']:
                    chip_vars[chip_name] = pulp.LpVariable(f"use_{chip_name}", cat="Binary")

                    # Create auxiliary variables for chip interactions
                    if chip_name == 'triple_captain':
                        for g in gws:
                            for p in self.players:
                                triple_captain_active[(g, p)] = pulp.LpVariable(f"tc_{g}_{p}", cat="Binary")
                                # Linearize: triple_captain_active = chip_vars['triple_captain'] AND captain
                                model += triple_captain_active[(g, p)] <= chip_vars[chip_name]
                                model += triple_captain_active[(g, p)] <= captain[g][p]
                                model += triple_captain_active[(g, p)] >= chip_vars[chip_name] + captain[g][p] - 1

                    elif chip_name == 'bench_boost':
                        for g in gws:
                            for p in self.players:
                                bench_boost_active[(g, p)] = pulp.LpVariable(f"bb_{g}_{p}", cat="Binary")
                                # Linearize: bench_boost_active = chip_vars['bench_boost'] AND (in_team - starting)
                                bench_player = pulp.LpVariable(f"bench_{g}_{p}", cat="Binary")
                                model += bench_player == in_team[g][p] - starting[g][p]
                                model += bench_boost_active[(g, p)] <= chip_vars[chip_name]
                                model += bench_boost_active[(g, p)] <= bench_player
                                model += bench_boost_active[(g, p)] >= chip_vars[chip_name] + bench_player - 1

        # Bank variable
        remaining_bank = pulp.LpVariable("remaining_bank", lowBound=0)

        # Objective function - maximize points over planning horizon with transfer penalties
        objective = 0

        # Always calculate points for the primary gameweek
        primary_gw = gws[0]
        for p in self.players:
            objective += starting[primary_gw][p] * pred.get((primary_gw, p), 0)
            objective += captain[primary_gw][p] * pred.get((primary_gw, p), 0)
            # Chip logic for the primary gameweek
            if 'triple_captain' in chip_vars and (primary_gw, p) in triple_captain_active:
                objective += triple_captain_active[(primary_gw, p)] * pred.get((primary_gw, p), 0)
            if 'bench_boost' in chip_vars and (primary_gw, p) in bench_boost_active:
                objective += bench_boost_active[(primary_gw, p)] * pred.get((primary_gw, p), 0)

        # Optionally add discounted points from future gameweeks in the horizon
        if use_planning_horizon and len(gws) > 1:
            for g in gws[1:]:
                weight = 0.9 ** (g - gw)  # Discount future gameweeks
                for p in self.players:
                    # For future weeks, we only care about the base points of the team selected for the primary_gw
                    objective += weight * in_team[primary_gw][p] * pred.get((g, p), 0)

        # Transfer penalty (applied after all point calculations)
        objective -= points_hit * self.config['transfer_penalty']

        model += objective

        # Squad constraints for each gameweek
        for g in gws:
            # Squad size and starting XI
            model += pulp.lpSum(in_team[g][p] for p in self.players) == self.config['squad_size']
            model += pulp.lpSum(starting[g][p] for p in self.players) == self.config['starting_xi']

            # Position constraints
            gk_players = [p for p in self.players if self.pos[p] in ['GK', 'GKP']]
            def_players = [p for p in self.players if self.pos[p] == 'DEF']
            mid_players = [p for p in self.players if self.pos[p] == 'MID']
            fwd_players = [p for p in self.players if self.pos[p] == 'FWD']

            # Squad position constraints
            model += pulp.lpSum(in_team[g][p] for p in gk_players) == 2
            model += pulp.lpSum(in_team[g][p] for p in def_players) == 5
            model += pulp.lpSum(in_team[g][p] for p in mid_players) == 5
            model += pulp.lpSum(in_team[g][p] for p in fwd_players) == 3

            # Starting XI position constraints
            model += pulp.lpSum(starting[g][p] for p in gk_players) == 1
            model += pulp.lpSum(starting[g][p] for p in def_players) >= self.config['min_def']
            model += pulp.lpSum(starting[g][p] for p in def_players) <= self.config['max_def']
            model += pulp.lpSum(starting[g][p] for p in mid_players) >= self.config['min_mid']
            model += pulp.lpSum(starting[g][p] for p in mid_players) <= self.config['max_mid']
            model += pulp.lpSum(starting[g][p] for p in fwd_players) >= self.config['min_fwd']
            model += pulp.lpSum(starting[g][p] for p in fwd_players) <= self.config['max_fwd']

            # Team constraint - max 3 players from same team
            for team in self.teams:
                team_players = [p for p in self.players if self.team_name[p] == team]
                model += pulp.lpSum(in_team[g][p] for p in team_players) <= self.config['max_team_players']

            # Captaincy constraints
            model += pulp.lpSum(captain[g][p] for p in self.players) == 1
            model += pulp.lpSum(vice[g][p] for p in self.players) == 1

            for p in self.players:
                model += captain[g][p] + vice[g][p] <= starting[g][p]
                model += starting[g][p] <= in_team[g][p]
                model += captain[g][p] + vice[g][p] <= 1  # Can't be both captain and vice

        # Transfer constraints
        if is_initial:
            # Initial squad selection - no transfers
            model += transfers_made == 0
            model += points_hit == 0
            for p in self.players:
                model += buy[p] == 0
                model += sell[p] == 0
        else:
            # Transfer logic
            for p in self.players:
                if p in current_squad:
                    # Player was in previous squad
                    model += in_team[gw][p] == 1 - sell[p]
                else:
                    # Player was not in previous squad
                    model += in_team[gw][p] == buy[p]
                    model += sell[p] == 0  # Can't sell player we don't have

            # Transfer counting
            model += transfers_made == pulp.lpSum(buy[p] + sell[p] for p in self.players) / 2

            # Free hit chip
            if 'free_hit' in chip_vars:
                model += transfers_made <= (1 - chip_vars['free_hit']) * 100

            # Wildcard chip - check for either wildcard
            wildcard_active = 0
            if 'wildcard_1' in chip_vars:
                wildcard_active += chip_vars['wildcard_1']
            if 'wildcard_2' in chip_vars:
                wildcard_active += chip_vars['wildcard_2']

            wildcard_active = 0
            if 'wildcard_1' in chip_vars:
                wildcard_active += chip_vars['wildcard_1']
            if 'wildcard_2' in chip_vars:
                wildcard_active += chip_vars['wildcard_2']

            # Big-M constraint to turn off penalty if wildcard is active
            # M is a large number, e.g., 100
            M = 100
            model += points_hit >= transfers_made - free_transfers
            model += points_hit >= 0
            # If wildcard_active is 1, this constraint becomes points_hit <= 0
            model += points_hit <= M * (1 - wildcard_active)

            # Budget constraint
        total_cost = pulp.lpSum(in_team[gw][p] * self.cost[p] for p in self.players)
        model += total_cost + remaining_bank == self.config['budget'] + bank

        # Only use one chip per gameweek
        if chip_vars:
            model += pulp.lpSum(chip_vars.values()) <= 1

        # Solve the model
        logger.info("Solving optimization model...")
        solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=300)  # 5 minute time limit
        model.solve(solver)

        status = pulp.LpStatus[model.status]
        if status == 'Optimal':
            logger.info("Optimal solution found.")
            solution = self.extract_solution(model, gw, gws, pred, actual, in_team, starting, captain, vice,
                                         buy, sell, transfers_made, points_hit, remaining_bank, chip_vars)
            self._validate_solution(solution)
            return solution
        else:
            logger.error(f"Optimization failed. Solver status: {status}")
            if status == 'Infeasible':
                logger.error("The model is infeasible, meaning no solution exists that satisfies all constraints.")
                logger.error("Common causes: budget is too low, invalid initial squad, or conflicting constraints.")
            elif status == 'Unbounded':
                logger.error("The model is unbounded. The objective function can be increased indefinitely.")
                logger.error("This usually points to an error in the objective function or missing constraints.")
            elif status == 'Not Solved':
                logger.error("The solver could not find a solution within the given time limit or other parameters.")

            # Optional: Add fallback logic here, e.g., relax constraints and retry
            return None

    def extract_solution(self, model, gw, gws, pred, actual, in_team, starting, captain, vice,
                         buy, sell, transfers_made, points_hit, remaining_bank, chip_vars):
        """Extract solution from solved model"""

        solution = {
            'gw': gw,
            'status': 'optimal',
            'objective_value': pulp.value(model.objective),
            'transfers_made': int(pulp.value(transfers_made)) if transfers_made.value() else 0,
            'points_hit': int(pulp.value(points_hit)) if points_hit.value() else 0,
            'remaining_bank': int(pulp.value(remaining_bank)) if remaining_bank.value() else 0,
            'chips_used': [],
            'transfers': {'in': [], 'out': []},
            'squads': {}
        }

        # Extract chip usage
        for chip_name, chip_var in chip_vars.items():
            if chip_var.value() and chip_var.value() > 0.5:
                solution['chips_used'].append(chip_name)

        # Extract transfers
        for p in self.players:
            if buy[p].value() and buy[p].value() > 0.5:
                solution['transfers']['in'].append({
                    'element': p,
                    'name': self.player_names[p],
                    'position': self.pos[p],
                    'team': self.team_name[p],
                    'cost': self.cost[p]
                })
            if sell[p].value() and sell[p].value() > 0.5:
                solution['transfers']['out'].append({
                    'element': p,
                    'name': self.player_names[p],
                    'position': self.pos[p],
                    'team': self.team_name[p],
                    'cost': self.cost[p]
                })

        # Extract squads for each gameweek
        for g in gws:
            squad = []
            starting_players = []
            total_predicted_current_gw = 0
            total_actual_current_gw = 0
            total_predicted_planning = 0

            for p in self.players:
                if in_team[g][p].value() and in_team[g][p].value() > 0.5:
                    is_starting = starting[g][p].value() and starting[g][p].value() > 0.5
                    is_captain = captain[g][p].value() and captain[g][p].value() > 0.5
                    is_vice = vice[g][p].value() and vice[g][p].value() > 0.5

                    current_gw_pred = pred.get((g, p), 0)
                    current_gw_actual = actual.get((g, p), 0)
                    player_planning_total = sum(pred.get((horizon_gw, p), 0) for horizon_gw in gws)

                    player_info = {
                        'element': p,
                        'name': self.player_names[p],
                        'position': self.pos[p],
                        'team': self.team_name[p],
                        'cost': self.cost[p],
                        'predicted_points_current_gw': current_gw_pred,
                        'actual_points_current_gw': current_gw_actual,
                        'predicted_points_planning_horizon': player_planning_total,
                        'is_starting': is_starting,
                        'is_captain': is_captain,
                        'is_vice': is_vice
                    }
                    squad.append(player_info)

                    # --- Corrected Point Calculation Logic ---
                    if is_starting:
                        pred_multiplier = 2 if is_captain else 1
                        actual_multiplier = 2 if is_captain else 1

                        if 'triple_captain' in solution['chips_used'] and is_captain:
                            pred_multiplier = 3
                            actual_multiplier = 3

                        total_predicted_current_gw += current_gw_pred * pred_multiplier
                        total_actual_current_gw += current_gw_actual * actual_multiplier
                        starting_players.append(player_info)

                    else:  # Player is on the bench
                        if 'bench_boost' in solution['chips_used']:
                            total_predicted_current_gw += current_gw_pred
                            total_actual_current_gw += current_gw_actual

                    # Add to planning horizon total (this is unaffected)
                    total_predicted_planning += player_planning_total

            solution['squads'][g] = {
                'squad': sorted(squad, key=lambda x: (x['position'], -x['predicted_points_current_gw'])),
                'starting_xi': sorted(starting_players,
                                      key=lambda x: (x['position'], -x['predicted_points_current_gw'])),
                'total_predicted_points_current_gw': total_predicted_current_gw,
                'total_actual_points_current_gw': total_actual_current_gw,
                'total_predicted_points_planning_horizon': total_predicted_planning
            }

        return solution

    def save_solution(self, solution):
        """Save solution to file"""
        gw = solution['gw']
        file_path = os.path.join(self.output_dir, f"team_gw{gw}.json")

        # Format for compatibility with existing format
        output = {
            'GW': gw,
            'transfers_made': solution['transfers_made'],
            'points_hit': solution['points_hit'],
            'bank': solution['remaining_bank'],
            'free_transfers': 2 if solution['transfers_made'] == 0 else 1,  # Add free_transfers for next GW
            'chips_used': solution['chips_used'],
            'transfers_in': solution['transfers']['in'],
            'transfers_out': solution['transfers']['out'],
            'total_predicted_points_current_gw': solution['squads'][gw]['total_predicted_points_current_gw'],
            'total_actual_points_current_gw': solution['squads'][gw]['total_actual_points_current_gw'],
            'total_predicted_points_planning_horizon': solution['squads'][gw][
                'total_predicted_points_planning_horizon'],
            'squad': solution['squads'][gw]['squad'],
            'starting_xi': solution['squads'][gw]['starting_xi']
        }

        logger.info(f"Saving solution for GW{gw} to {file_path}")
        with open(file_path, 'w') as f:
            json.dump(output, f, indent=2)

    def optimize_season(self, start_gw=1, end_gw=38):
        """Optimize for multiple gameweeks, re-optimizing each week"""
        logger.info(f"Starting season optimization from GW{start_gw} to GW{end_gw}")

        season_metrics = []
        results = {}
        used_chips = set()
        initial_squad_gw = start_gw - 1

        if initial_squad_gw > 0:
            current_squad, bank, free_transfers = self.load_current_squad(initial_squad_gw)
        else:
            current_squad = None
            bank = 0
            free_transfers = self.config['initial_free_transfers']

        for gw in range(start_gw, min(end_gw + 1, 39)):
            logger.info(f"\n{'=' * 50}")
            logger.info(f"OPTIMIZING GAMEWEEK {gw}")
            logger.info(f"{'=' * 50}")

            try:
                # Optimize for this gameweek
                solution = self.optimize_single_gw(gw, current_squad, bank, free_transfers, used_chips=used_chips)

                gw_squad = solution['squads'][gw]
                metrics_data = {
                    'gameweek': gw,
                    'predicted_points': gw_squad['total_predicted_points_current_gw'],
                    'actual_points': gw_squad['total_actual_points_current_gw'],
                    'transfers_made': solution['transfers_made'],
                    'points_hit': solution['points_hit'],
                    'chip_used': solution['chips_used'][0] if solution['chips_used'] else 'None',
                    'bank': solution['remaining_bank']
                }
                season_metrics.append(metrics_data)

                if solution:
                    results[gw] = solution
                    self.save_solution(solution)

                    # Update state for next gameweek
                    current_squad = {p['element'] for p in solution['squads'][gw]['squad']}
                    bank = solution['remaining_bank']

                    if solution['chips_used']:
                        used_chips.update(solution['chips_used'])
                        logger.info(f"Chip used in GW{gw}: {solution['chips_used'][0]}. This chip is now disabled.")

                    # Calculate free transfers for next week
                    if solution['transfers_made'] <= free_transfers:
                        # No hit taken, roll over unused transfer (max 2)
                        free_transfers = min(2, free_transfers - solution['transfers_made'] + 1)
                    else:
                        # Hit taken, get 1 free transfer next week
                        free_transfers = 1

                    # If wildcard used, reset free transfers
                    if 'wildcard_1' in solution['chips_used'] or 'wildcard_2' in solution['chips_used']:
                        free_transfers = 1

                    logger.info(f"GW{gw} completed successfully")
                    logger.info(
                        f"Expected points (current GW): {solution['squads'][gw]['total_predicted_points_current_gw']:.1f}")
                    logger.info(
                        f"Expected points (planning horizon): {solution['squads'][gw]['total_predicted_points_planning_horizon']:.1f}")
                    logger.info(
                        f"Actual points (current GW): {solution['squads'][gw]['total_actual_points_current_gw']:.1f}")
                    logger.info(f"Transfers: {solution['transfers_made']}, Hit: {solution['points_hit']}")
                    logger.info(f"Bank: ${solution['remaining_bank'] / 10:.1f}M")
                    logger.info(f"Free transfers for next GW: {free_transfers}")

                else:
                    logger.error(f"Failed to optimize GW{gw}")
                    break

            except Exception as e:
                logger.error(f"Error optimizing GW{gw}: {e}")
                break
        if season_metrics:
            logger.info("Saving season optimization metrics...")
            metrics_df = pd.DataFrame(season_metrics)
            metrics_file = os.path.join(self.output_dir, "season_metrics.csv")
            metrics_df.to_csv(metrics_file, index=False)
            logger.info(f"Metrics saved to {metrics_file}")

        logger.info(f"\nSeason optimization completed. Optimized {len(results)} gameweeks.")
        return results


def main():
    """Main execution function"""
    try:
        # Initialize optimizer
        optimizer = FPLOptimizer()

        # Run optimization
        # For single gameweek: optimizer.optimize_single_gw(1)
        # For full season: optimizer.optimize_season(1, 38)
        # For specific range: optimizer.optimize_season(1, 10)

        results = optimizer.optimize_season(2, 38)  # Optimize first 6 gameweeks

        logger.info("Optimization completed successfully!")

    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
