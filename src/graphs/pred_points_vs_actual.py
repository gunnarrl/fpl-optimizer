import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def create_scatterplot(csv_path: str):
    """
    Loads FPL prediction data, creates a scatterplot of predicted vs.
    actual points, calculates the R-squared value, and colors points
    by gameweek.

    Args:
        csv_path (str): The file path to the CSV data.
    """
    try:
        # Load the dataset from the specified path
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file was not found at {csv_path}")
        print("Please ensure the path is correct relative to where you are running the script.")
        return

    if df.empty:
        print(f"No data found in the CSV file at {csv_path}.")
        return

    # Ensure the columns exist before proceeding
    required_cols = ['predicted_points_gw+1', 'points_gw+1', 'GW']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: The CSV must contain the columns: {', '.join(required_cols)}")
        return

    # Drop rows with missing values in the relevant columns to ensure r2_score works correctly
    df.dropna(subset=required_cols, inplace=True)

    # Calculate R-squared value for the entire dataset
    # r2_score(y_true, y_pred)
    r2 = r2_score(df['points_gw+1'], df['predicted_points_gw+1'])

    # Set the aesthetic style of the plots
    sns.set_theme(style="whitegrid")

    # Create the scatterplot using the entire dataframe
    # The 'hue' parameter automatically colors the points by the 'GW' column.
    # The 'palette' parameter provides a visually appealing color scheme.
    plt.figure(figsize=(12, 8))
    scatter_plot = sns.scatterplot(
        data=df,
        x='predicted_points_gw+1',
        y='points_gw+1',
        hue='GW',
        palette='viridis',  # A nice color map for sequential data like gameweeks
        s=50  # marker size
    )

    # Set the title and labels for clarity, including the R-squared value
    plt.title(f'Predicted vs. Actual Points by Gameweek\nOverall $R^2 = {r2:.3f}$', fontsize=16)
    plt.xlabel('Predicted Points (predicted_points_gw+1)', fontsize=12)
    plt.ylabel('Actual Points (points_gw+1)', fontsize=12)

    # Move the legend outside of the plot area to prevent overlap
    scatter_plot.legend(title='Gameweek', bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    # Ensure the plot layout is tight
    plt.tight_layout()

    # Display the plot
    plt.show()


if __name__ == '__main__':
    # Define the path to your CSV file.
    # The path is relative to the location of the script.
    # You might need to adjust it based on your project structure.
    CSV_FILE_PATH = '../../data/output/predictions/koa_predictions_updated.csv'

    # Call the function to create and display the plot
    create_scatterplot(CSV_FILE_PATH)
