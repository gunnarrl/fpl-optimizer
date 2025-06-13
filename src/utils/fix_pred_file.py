import pandas as pd

def update_player_data(predictions_file, fpl_data_file, output_file):
    """
    Updates 'element' and 'position' columns in a predictions CSV
    based on a master FPL data CSV for the '2024-25' season.

    Args:
        predictions_file (str): Filepath for the predictions CSV.
        fpl_data_file (str): Filepath for the processed FPL data CSV.
        output_file (str): Filepath for the updated output CSV.
    """
    try:
        # Load the datasets from the provided file paths
        predictions_df = pd.read_csv(predictions_file)
        fpl_data_df = pd.read_csv(fpl_data_file)

        # --- Step 1: Filter FPL data for the correct season ---
        # We only want to use player data from the '2024-25' season as the source of truth.
        fpl_season_df = fpl_data_df[fpl_data_df['season_x'] == '2024-25'].copy()

        # --- Step 2: Create a mapping from name to position and element ---
        # We'll set the 'name' column as the index to easily look up players.
        # This is more efficient than iterating through the DataFrame for every player.
        player_map = fpl_season_df.set_index('name')

        # Create dictionaries for mapping positions and elements
        position_map = player_map['position'].to_dict()
        element_map = player_map['element'].to_dict()

        # --- Step 3: Update the predictions DataFrame ---
        # Use the .map() method to update the columns.
        # It will look up each name from the predictions file in our new mapping dictionaries.
        # 'na_action='ignore'' ensures that if a player isn't found in the map,
        # their original value is kept instead of being replaced with NaN (Not a Number).
        predictions_df['position'] = predictions_df['name'].map(position_map).fillna(predictions_df['position'])
        predictions_df['element'] = predictions_df['name'].map(element_map).fillna(predictions_df['element'])


        # --- Step 4: Save the updated data to a new file ---
        predictions_df.to_csv(output_file, index=False)

        print(f"Successfully updated player data and saved to '{output_file}'")
        print("\n--- First 5 rows of the updated file ---")
        print(predictions_df.head())

    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure the file paths are correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

# --- Define your file paths here ---
# Note: The ../../ indicates going up two directories. Adjust as needed for your folder structure.
predictions_path = '../../data/output/predictions/koa_predictions.csv'
fpl_data_path = '../../data/processed/processed_fpl_data.csv'
output_path = '../../data/output/predictions/koa_predictions_updated.csv'  # This will save in the same directory as the script

# --- Run the function ---
update_player_data(predictions_path, fpl_data_path, output_path)