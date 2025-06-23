import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def plot_player_performance_combined(df: pd.DataFrame, player_elements: list, save_path: str = None):
    """
    Generates a single, combined line graph showing predicted points and actual points
    trending over gameweeks for multiple players.

    Args:
        df (pd.DataFrame): The DataFrame containing player data.
                           Expected columns: 'name', 'element', 'GW',
                           'predicted_points_gw+1', 'points_gw+1'.
        player_elements (list): A list of integer player 'element' IDs to graph.
        save_path (str, optional): The file path to save the plot (e.g., 'player_stats.png').
                                   If None, the plot is displayed. Defaults to None.
    """
    # --- 1. Data Preparation ---

    # Filter the DataFrame for the specified players
    df['element'] = pd.to_numeric(df['element'], errors='coerce')
    filtered_df = df[df['element'].isin(player_elements)].copy()
    filtered_df.dropna(subset=['element'], inplace=True)

    # Convert 'gameweek' to a numeric type for correct plotting
    filtered_df['GW'] = pd.to_numeric(filtered_df['GW'], errors='coerce')

    # Create a mapping from player ID to name for clearer labels
    # You can expand this map with more players from your dataset
    player_names_map = {
        328: "Mohamed Salah",
        351: "Erling Haaland",
        182: "Cole Palmer"
    }
    filtered_df['name'] = filtered_df['element'].map(player_names_map).fillna(filtered_df['name'])

    # Sort data for proper line plotting
    filtered_df.sort_values(by=['name', 'GW'], inplace=True)

    # "Melt" the DataFrame to transform it into a long format suitable for seaborn.
    # This creates a single 'Metric' column (for predicted vs. actual points) and
    # a 'Points' column for their values, which allows plotting them together.
    melted_df = filtered_df.melt(
        id_vars=['GW', 'name'],
        value_vars=['predicted_points_gw+1', 'points_gw+1'],
        var_name='Metric',
        value_name='Points'
    )

    # Make the legend labels more friendly
    melted_df['Metric'] = melted_df['Metric'].replace({
        'predicted_points_gw+1': 'Predicted Points',
        'points_gw+1': 'Actual Points'
    })

    # --- 2. Plotting ---

    # Set the visual style for the plot
    sns.set_theme(style="whitegrid", palette="deep")

    # Create the figure and a single subplot (axes)
    fig, ax = plt.subplots(figsize=(16, 9))

    # Generate the line plot.
    # 'hue' will use a different color for each player ('name').
    # 'style' will use a different line style for each 'Metric' (Predicted vs. Actual).
    sns.lineplot(
        data=melted_df,
        x='GW',
        y='Points',
        hue='name',
        style='Metric',
        marker='o',
        markersize=6,
        ax=ax,
        linewidth=2.5
    )

    # --- 3. Aesthetics and Customization ---

    # Set plot title and labels with improved font sizes
    ax.set_title('Player Performance: Predicted vs. Actual Points', fontsize=20, pad=20)
    ax.set_xlabel('Gameweek (GW)', fontsize=14)
    ax.set_ylabel('Points', fontsize=14)

    # Customize the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles, labels=labels, title='Player / Metric', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Set grid and ticks for better readability
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.xticks(rotation=0)  # Keep x-axis labels horizontal

    # Adjust layout to prevent the legend from being cut off
    plt.tight_layout(rect=[0, 0, 0.85, 1])

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

# 1. Load your CSV into a pandas DataFrame.
#    Make sure the path to your CSV file is correct.
try:
    df = pd.read_csv('../../data/output/predictions/koa_predictions_updated.csv')

    # 2. Define the player element IDs you want to visualize.
    #    These are the example IDs for Salah, Haaland, and Palmer.
    selected_player_elements = [328]

    # 3. Call the plotting function to generate and save the graph.
    plot_player_performance_combined(df, selected_player_elements, save_path='fpl_player_performance_combined.png')

except FileNotFoundError:
    print("Error: 'koa_predictions_updated.csv' not found.")
    print("Please make sure the CSV file is in the same directory as the script, or provide the full path.")

