import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error


def plot_salah_multi_gameweek_forecast(df: pd.DataFrame, player_element: int, save_path: str = None):
    """
    Generates a series of 6 subplots for a single player (Mohamed Salah) showing
    predicted_points_gw+{i} vs. points_gw+{i} for i=1 to 6, and calculates the MAE.

    Args:
        df (pd.DataFrame): The DataFrame containing player data.
        player_element (int): The integer player 'element' ID to graph.
        save_path (str, optional): The file path to save the plot.
                                   If None, the plot is displayed. Defaults to None.
    """
    # --- 1. Data Preparation ---

    # Filter the DataFrame for the specified player
    df['element'] = pd.to_numeric(df['element'], errors='coerce')
    player_data = df[df['element'] == player_element].copy()

    if player_data.empty:
        print(f"No data found for player element ID: {player_element}")
        return

    # Convert 'GW' to a numeric type for correct plotting
    player_data['GW'] = pd.to_numeric(player_data['GW'], errors='coerce')
    player_data.sort_values(by='GW', inplace=True)

    # Get player name for the title
    player_name = "Mohamed Salah"  # Hardcoded for this specific request

    # --- 2. Plotting & MAE Calculation ---

    # Set the visual style for the plot
    sns.set_theme(style="whitegrid", palette="deep")

    # Create the figure and a set of 6 subplots, sharing the x-axis
    fig, axes = plt.subplots(6, 1, figsize=(14, 28), sharex=True)

    # Add a main title for the entire figure
    fig.suptitle(f'{player_name} - Multi-Gameweek Point Forecast vs. Actual Performance', fontsize=22, y=1.0)

    # Loop through each forecast horizon (GW+1 to GW+6)
    for i in range(1, 7):
        ax = axes[i - 1]  # Get the current subplot axis

        # Define the column names for this subplot
        pred_col = f'predicted_points_gw+{i}'
        actual_col = f'points_gw+{i}'

        # Verify that the necessary columns exist in the DataFrame
        if pred_col not in player_data.columns or actual_col not in player_data.columns:
            ax.text(0.5, 0.5, f'Data for GW+{i} not available', ha='center', va='center', fontsize=12)
            ax.set_title(f'Forecast Horizon: Gameweek +{i}', fontsize=16, pad=10)
            continue

        # --- MAE CALCULATION ---
        # Drop rows where either prediction or actual is NaN to ensure a fair comparison
        mae_data = player_data[[pred_col, actual_col]].dropna()
        mae_score = 0
        if not mae_data.empty:
            mae_score = mean_absolute_error(mae_data[actual_col], mae_data[pred_col])

        # --- PLOTTING ---
        # Plot Predicted Points
        sns.lineplot(
            data=player_data,
            x='GW',
            y=pred_col,
            marker='o',
            markersize=7,
            linewidth=2.5,
            label=f'Predicted Points (GW+{i})',
            ax=ax,
            color='#1f77b4'  # A nice blue
        )

        # Plot Actual Points
        sns.lineplot(
            data=player_data,
            x='GW',
            y=actual_col,
            marker='x',
            markersize=7,
            linewidth=2.5,
            linestyle='--',
            label=f'Actual Points (GW+{i})',
            ax=ax,
            color='#ff7f0e'  # A distinct orange
        )

        # --- 3. Aesthetics and Customization for each subplot ---
        # Update title to include the calculated MAE score
        ax.set_title(f'Forecast Horizon: Gameweek +{i} (MAE: {mae_score:.2f})', fontsize=16, pad=10)
        ax.set_ylabel('Points', fontsize=14)
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.tick_params(axis='y', labelsize=12)

    # Set the common X-axis label only on the bottom-most subplot
    axes[-1].set_xlabel('Gameweek of Prediction (GW)', fontsize=14)
    axes[-1].tick_params(axis='x', labelsize=12)

    # Find overall min/max gameweeks for setting x-axis limits consistently
    gw_min = player_data['GW'].min()
    gw_max = player_data['GW'].max()
    plt.setp(axes, xlim=(gw_min - 1, gw_max + 1))

    # Adjust layout to prevent overlapping titles/labels and fit the main title
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    # --- 4. Save or Display the Plot ---
    if save_path:
        try:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Plot saved successfully to {save_path}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    else:
        plt.show()


# --- How to use this function ---

# You may need to install scikit-learn: pip install scikit-learn

# 1. Load your CSV into a pandas DataFrame.
#    Ensure the path to your CSV file is correct.
try:
    df = pd.read_csv('../../data/output/predictions/koa_predictions_updated.csv')

    # 2. Define the player element ID for Mohamed Salah.
    salah_element_id = 328

    # 3. Call the plotting function to generate and save the graph.
    plot_salah_multi_gameweek_forecast(
        df,
        salah_element_id,
        save_path='salah_multi_gw_forecast_with_mae.png'
    )

except FileNotFoundError:
    print("Error: 'koa_predictions_updated.csv' not found.")
    print("Please make sure the CSV file is in the same directory as the script, or provide the full path.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
