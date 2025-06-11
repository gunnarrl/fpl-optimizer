import pandas as pd
import os


def add_expected_points(master_filepath, output_filepath):
    """
    Loads a master dataset, finds and combines seasonal expected points (xP) data,
    handles duplicates in the source data, and merges it into the master dataset
    using 'element' and 'kickoff_time' as unique keys.

    Args:
        master_filepath (str): The path to the master CSV file (e.g., master_combined.csv).
        output_filepath (str): The path to save the final CSV with the xP column.
    """
    # --- 1. Load the Master Dataset ---
    try:
        master_df = pd.read_csv(master_filepath)
        print(f"Successfully loaded master dataset from '{master_filepath}'. Shape: {master_df.shape}")
        original_row_count = len(master_df)
    except FileNotFoundError:
        print(f"Error: The master file was not found at '{master_filepath}'. Please check the path.")
        return
    except Exception as e:
        print(f"An error occurred while loading the master file: {e}")
        return

    # --- 2. Find, Load, and Combine Seasonal xP Data ---
    xp_data_frames = []
    base_raw_data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw')

    print("\n--- Searching for and loading Expected Points (xP) data ---")

    for year in range(2020, 2024):
        season_folder_name = f"{year}-{(year + 1) % 100:02d}"
        xp_file_path = os.path.join(base_raw_data_path, season_folder_name, 'gws', 'merged_gw.csv')

        if os.path.exists(xp_file_path):
            try:
                gw_df = pd.read_csv(xp_file_path)

                # Use 'element', 'kickoff_time', and 'xP' as the required columns
                required_columns = ['element', 'kickoff_time', 'xP']
                if all(col in gw_df.columns for col in required_columns):
                    temp_df = gw_df[required_columns].copy()
                    xp_data_frames.append(temp_df)
                    print(f"  > Successfully loaded and processed xP data for season {season_folder_name}")
                else:
                    missing_cols = [col for col in required_columns if col not in gw_df.columns]
                    print(
                        f"  > WARNING: Skipped file for season {season_folder_name}. Missing required columns: {', '.join(missing_cols)}.")
            except Exception as e:
                print(f"  > ERROR: Could not read or process file for season {season_folder_name}. Reason: {e}")
        else:
            print(f"  > INFO: No xP data file found for season {season_folder_name} at '{xp_file_path}'.")

    # --- 3. Merge the xP Data into the Master DataFrame ---
    if not xp_data_frames:
        print("\n--- WARNING: No xP data was found. The 'xP' column will not be added. ---")
    else:
        all_xp_df = pd.concat(xp_data_frames, ignore_index=True)
        print(f"\nSuccessfully combined all available xP data. Total xP entries before deduplication: {len(all_xp_df)}")

        all_xp_df.drop_duplicates(subset=['element', 'kickoff_time'], keep='last', inplace=True)
        print(f"Deduplication complete. Total unique xP entries to merge: {len(all_xp_df)}")

        # Ensure data types are consistent for merging
        master_df['kickoff_time'] = pd.to_datetime(master_df['kickoff_time'])
        all_xp_df['kickoff_time'] = pd.to_datetime(all_xp_df['kickoff_time'])
        master_df['element'] = master_df['element'].astype(int)
        all_xp_df['element'] = all_xp_df['element'].astype(int)

        # Perform a left merge to add the xP column to the master DataFrame
        master_df = pd.merge(master_df, all_xp_df, on=['element', 'kickoff_time'], how='left')
        print("Merge complete. 'xP' column has been added to the master dataset.")

        # --- 4. Handle Missing Values ---
        master_df['xP'] = master_df['xP'].fillna(0.0)
        print("Filled missing 'xP' values with 0.")

    # --- 5. Validate Final Row Count and Save ---
    if len(master_df) != original_row_count:
        print(
            f"\n--- CRITICAL WARNING: The number of rows changed from {original_row_count} to {len(master_df)}. Please investigate the data. ---")
    else:
        print(f"\nRow count validation passed. The number of rows remains {original_row_count}.")

    try:
        output_dir = os.path.dirname(output_filepath)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        master_df.to_csv(output_filepath, index=False)
        print(f"\nSuccessfully saved the updated master dataset to '{output_filepath}'.")
        print(f"Final dataset shape: {master_df.shape}")

    except Exception as e:
        print(f"\nError: Could not save the final file. Reason: {e}")


if __name__ == '__main__':
    master_file = '../../data/raw/cleaned/master_combined.csv'
    output_file_with_xp = '../../data/raw/cleaned/master_combined_with_xp.csv'

    add_expected_points(master_file, output_file_with_xp)