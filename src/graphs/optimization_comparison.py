import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from collections import Counter


def load_data(autogluon_path, xgboost_path):
    """
    Loads the season metrics CSV files for both models into pandas DataFrames.

    Args:
        autogluon_path (str): File path for the Autogluon data.
        xgboost_path (str): File path for the XGBoost data.

    Returns:
        tuple: A tuple containing the Autogluon DataFrame and the XGBoost DataFrame.
        Returns (None, None) if a file is not found.
    """
    try:
        autogluon_df = pd.read_csv(autogluon_path)
        xgboost_df = pd.read_csv(xgboost_path)
        print("Successfully loaded both CSV files.")
        return autogluon_df, xgboost_df
    except FileNotFoundError as e:
        print(f"Error: File not found - {e}. Please ensure the file paths are correct.")
        print("Autogluon path provided:", autogluon_path)
        print("XGBoost path provided:", xgboost_path)
        return None, None


def calculate_summary_statistics(df, model_name):
    """
    Calculates and returns summary statistics for a given model's DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing the season data.
        model_name (str): The name of the model for labeling.

    Returns:
        dict: A dictionary containing the summary statistics.
    """
    stats = {}
    stats['Model'] = model_name
    stats['Total Net Points'] = df['actual_points'].sum() - df['points_hit'].sum()
    stats['Mean Squared Prediction Error'] = np.mean((df['predicted_points'] - df['actual_points']) ** 2)
    # NEW: Calculate prediction accuracy as (total actual / total predicted)
    stats['Prediction Accuracy (%)'] = (df['actual_points'].sum() / df['predicted_points'].sum()) * 100
    # NEW: Calculate average bank value
    stats['Average Bank Value'] = df['bank'].mean()

    # Also return these for context if needed, though not for the main table
    stats['Total Transfers Made'] = df['transfers_made'].sum()
    stats['Total Points Hit'] = df['points_hit'].sum()

    return stats


def analyze_team_similarity(ag_base_path, xgb_base_path):
    """
    Analyzes squad similarity and finds the most common players between models.

    Args:
        ag_base_path (str): The base directory path for Autogluon's JSON team data.
        xgb_base_path (str): The base directory path for XGBoost's JSON team data.

    Returns:
        tuple: A tuple containing:
            - dict: Gameweeks as keys, number of common players as values.
            - collections.Counter: A counter of player IDs for common players.
            - dict: A mapping of player IDs to player names.
    """
    similarity_data = {}
    common_player_counter = Counter()
    player_id_to_name = {}
    print("\n--- Analyzing Team Similarity ---")

    for gw in range(1, 39):
        ag_json_path = os.path.join(ag_base_path, f'team_gw{gw}.json')
        xgb_json_path = os.path.join(xgb_base_path, f'team_gw{gw}.json')

        try:
            with open(ag_json_path, 'r') as f:
                ag_data = json.load(f)
            with open(xgb_json_path, 'r') as f:
                xgb_data = json.load(f)

            ag_squad_ids = {player['element'] for player in ag_data.get('squad', [])}
            xgb_squad_ids = {player['element'] for player in xgb_data.get('squad', [])}

            for player in ag_data.get('squad', []) + xgb_data.get('squad', []):
                if 'element' in player and 'name' in player:
                    player_id_to_name[player['element']] = player['name']

            if not ag_squad_ids or not xgb_squad_ids:
                continue

            common_player_ids = ag_squad_ids.intersection(xgb_squad_ids)
            similarity_data[gw] = len(common_player_ids)
            common_player_counter.update(common_player_ids)

        except FileNotFoundError:
            break
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not process JSON for Gameweek {gw}. Error: {e}. Skipping.")
            continue

    print(f"Completed similarity analysis for {len(similarity_data)} gameweeks.")
    return similarity_data, common_player_counter, player_id_to_name


def plot_cumulative_difference(ag_df, xgb_df, output_dir):
    """
    Plots the cumulative difference in net points between the two models.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ag_df['cumulative_net_points'] = (ag_df['actual_points'] - ag_df['points_hit']).cumsum()
    xgb_df['cumulative_net_points'] = (xgb_df['actual_points'] - xgb_df['points_hit']).cumsum()

    point_difference = ag_df['cumulative_net_points'] - xgb_df['cumulative_net_points']

    ax.plot(ag_df['gameweek'], point_difference, marker='o', linestyle='-', markersize=4, label='Point Advantage')
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.fill_between(ag_df['gameweek'], point_difference, 0, where=point_difference >= 0, facecolor='green', alpha=0.3,
                    interpolate=True, label='Autogluon Leading')
    ax.fill_between(ag_df['gameweek'], point_difference, 0, where=point_difference < 0, facecolor='red', alpha=0.3,
                    interpolate=True, label='XGBoost Leading')

    ax.set_title('Cumulative Point Advantage: Autogluon vs. XGBoost', fontsize=16, weight='bold')
    ax.set_xlabel('Gameweek', fontsize=12)
    ax.set_ylabel('Point Difference (Autogluon - XGBoost)', fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cumulative_difference_comparison.png'))
    plt.close()
    print("Saved cumulative difference plot to:", output_dir)


def plot_bank_value(ag_df, xgb_df, output_dir):
    """
    NEW: Plots the bank value over the season for both models.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    ax.plot(ag_df['gameweek'], ag_df['bank'], marker='.', linestyle='-', label='Autogluon Bank Value')
    ax.plot(xgb_df['gameweek'], xgb_df['bank'], marker='.', linestyle='--', label='XGBoost Bank Value')

    ax.set_title('Bank Value Over the Season', fontsize=16, weight='bold')
    ax.set_xlabel('Gameweek', fontsize=12)
    ax.set_ylabel('Bank Value (£M)', fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'bank_value_comparison.png'))
    plt.close()
    print("Saved bank value comparison plot to:", output_dir)


if __name__ == "__main__":
    # --- Configuration ---
    AUTOGLUON_PATH = '../../data/output/teams/autoGluon/season_metrics.csv'
    XGBOOST_PATH = '../../data/output/teams/xgb/season_metrics.csv'
    AUTOGLUON_JSON_BASE_PATH = '../../data/output/teams/autoGluon/'
    XGBOOST_JSON_BASE_PATH = '../../data/output/teams/xgb/'
    OUTPUT_DIR = 'fpl_analysis_plots'

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- Main Execution ---
    autogluon_data, xgboost_data = load_data(AUTOGLUON_PATH, XGBOOST_PATH)

    if autogluon_data is not None and xgboost_data is not None:
        ag_summary = calculate_summary_statistics(autogluon_data, 'Autogluon')
        xgb_summary = calculate_summary_statistics(xgboost_data, 'XGBoost')

        # Create a clean DataFrame for the final table
        summary_df = pd.DataFrame([ag_summary, xgb_summary])
        display_df = summary_df[
            ['Model', 'Total Net Points', 'Mean Squared Prediction Error', 'Prediction Accuracy (%)',
             'Average Bank Value']].copy()
        display_df.rename(columns={'Mean Squared Prediction Error': 'Prediction Error (MSE)'}, inplace=True)

        print("\n--- Performance Summary ---")
        # Format the numbers for better readability in the console
        display_df['Total Net Points'] = display_df['Total Net Points'].map('{:,.2f}'.format)
        display_df['Prediction Error (MSE)'] = display_df['Prediction Error (MSE)'].map('{:,.2f}'.format)
        display_df['Prediction Accuracy (%)'] = display_df['Prediction Accuracy (%)'].map('{:,.2f}%'.format)
        display_df['Average Bank Value'] = display_df['Average Bank Value'].map('£{:,.2f}M'.format)
        print(display_df.to_string(index=False))

        # --- Generate and save all plots ---
        plot_cumulative_difference(autogluon_data.copy(), xgboost_data.copy(), OUTPUT_DIR)
        plot_bank_value(autogluon_data, xgboost_data, OUTPUT_DIR)  # NEW PLOT

        # --- Optional: Analyze team similarity ---
        similarity_results, player_counter, id_map = analyze_team_similarity(AUTOGLUON_JSON_BASE_PATH,
                                                                             XGBOOST_JSON_BASE_PATH)
        if similarity_results:
            avg_similarity = np.mean(list(similarity_results.values()))
            print(f"\nAverage number of common players per gameweek: {avg_similarity:.2f} / 15")

        print(f"\nAnalysis complete. All plots saved in the '{OUTPUT_DIR}' directory.")
