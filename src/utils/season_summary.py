import pandas as pd


def summarize_fpl_season(csv_filepath):
    """
    Reads FPL data from a CSV file and calculates a season summary.

    Args:
        csv_filepath (str): The path to the CSV file.

    Returns:
        None: Prints the summary to the console.
    """
    try:
        # Load the dataset from the specified CSV file
        df = pd.read_csv(csv_filepath)

        # --- Data Validation ---
        # Check if required columns exist
        required_columns = ['actual_points', 'points_hit', 'predicted_points', 'transfers_made', 'chip_used', 'bank']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: The CSV must contain the following columns: {required_columns}")
            return

        # --- Core Calculations ---
        # Calculate total actual points scored across all gameweeks
        total_actual_points = df['actual_points'].sum()

        # Calculate the total points spent on transfers (hits)
        total_points_hits = df['points_hit'].sum()

        # Calculate the final net total points
        net_total_points = total_actual_points - total_points_hits

        # --- Additional Summary Statistics ---
        # Calculate total predicted points for comparison
        total_predicted_points = df['predicted_points'].sum()

        # Sum the total number of transfers made over the season
        total_transfers = df['transfers_made'].sum()

        # Identify which chips were used (ignoring empty/NaN values)
        chips_used = df['chip_used'].dropna().unique().tolist()

        # Find the average bank value over the season
        average_bank = df['bank'].mean()

        # Find the highest scoring gameweek
        highest_scoring_gw = df.loc[df['actual_points'].idxmax()]

        # --- Display the Summary ---
        print("--- FPL Season Summary ---")
        print(f"\nTotal Net Points: {net_total_points}")
        print("==========================")
        print(f"Total Actual Points Scored: {total_actual_points}")
        print(f"Total Points on Hits: {total_points_hits}")

        print("\n--- Additional Insights ---")
        print(f"Total Predicted Points: {total_predicted_points}")
        print(f"Accuracy (Actual vs Predicted): {total_actual_points / total_predicted_points:.2%}")
        print(f"Total Transfers Made: {total_transfers}")
        print(f"Chips Used This Season: {', '.join(chips_used) if chips_used else 'None'}")
        print(f"Average Bank Value: £{average_bank:.2f}m")
        print(
            f"\nHighest Scoring Gameweek (GW{int(highest_scoring_gw['gameweek'])}): {highest_scoring_gw['actual_points']} points")

    except FileNotFoundError:
        print(f"Error: The file '{csv_filepath}' was not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- How to use the script ---
if __name__ == "__main__":
    # 1. Make sure your CSV file is in the same directory as this script,
    #    or provide the full path to the file.
    # 2. Replace 'your_fpl_data.csv' with the actual name of your file.
    csv_file_name = '../../data/output/teams/autoGluon/season_metrics.csv'

    summarize_fpl_season(csv_file_name)