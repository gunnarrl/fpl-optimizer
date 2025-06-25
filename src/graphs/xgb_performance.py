import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, roc_curve, auc, accuracy_score


def plot_performance_visuals(file_path):
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from sklearn.metrics import mean_absolute_error, roc_curve, auc, accuracy_score
    from scipy.stats import spearmanr

    def plot_performance_visuals(file_path):
        """
        Loads performance data and generates several visualizations to assess
        model performance.

        Args:
            file_path (str): The path to the CSV file containing the performance data.
        """
        try:
            # Load the dataset
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: The file at {file_path} was not found.")
            return

        # Set plot style
        sns.set_theme(style="whitegrid")

        # --- Create a combined figure with two subplots ---
        fig, axes = plt.subplots(2, 1, figsize=(7, 9))  # Width=7, Height=9 inches

        # --- Subplot 1: Predicted vs. Actual Points (GW+1) ---
        predicted_points_col = 'predicted_points_gw+1'
        actual_points_col = 'points_gw+1'

        scatter_df = df[[predicted_points_col, actual_points_col]].dropna()

        axes[0].scatter(scatter_df[predicted_points_col], scatter_df[actual_points_col], alpha=0.6, edgecolors='w',
                        s=50)
        perfect_line = np.linspace(min(scatter_df[actual_points_col].min(), scatter_df[predicted_points_col].min()),
                                   max(scatter_df[actual_points_col].max(), scatter_df[predicted_points_col].max()),
                                   100)
        axes[0].plot(perfect_line, perfect_line, 'r--', linewidth=2, label='Perfect Prediction')
        axes[0].set_title('Predicted vs. Actual Points (GW+1)', fontsize=14)
        axes[0].set_xlabel('Predicted Points', fontsize=10)
        axes[0].set_ylabel('Actual Points', fontsize=10)
        axes[0].grid(True, linestyle='--', alpha=0.6)
        axes[0].legend()

        # --- Subplot 2: Combined ROC AUC Curve for all Game Weeks ---
        all_roc_auc = []
        all_class_accuracies = []
        played_accuracies = []

        for i in range(1, 7):
            actual_minutes_col_gw = f'actual_minutes_gw+{i}'
            predicted_prob_col_gw = f'predicted_prob_gte_60_mins_gw+{i}'

            if actual_minutes_col_gw not in df.columns or predicted_prob_col_gw not in df.columns:
                continue

            temp_df = df[[actual_minutes_col_gw, predicted_prob_col_gw]].dropna()
            if temp_df.empty or len(np.unique((temp_df[actual_minutes_col_gw] >= 60))) < 2:
                continue

            y_true_roc = (temp_df[actual_minutes_col_gw] >= 60).astype(int)
            y_scores = temp_df[predicted_prob_col_gw]

            fpr, tpr, _ = roc_curve(y_true_roc, y_scores)
            roc_auc = auc(fpr, tpr)
            all_roc_auc.append(roc_auc)

            y_pred_class = (y_scores > 0.5).astype(int)
            all_class_accuracies.append(accuracy_score(y_true_roc, y_pred_class))

            played_df = temp_df[temp_df[actual_minutes_col_gw] > 0]
            if not played_df.empty and len(np.unique((played_df[actual_minutes_col_gw] >= 60))) > 1:
                y_true_played = (played_df[actual_minutes_col_gw] >= 60).astype(int)
                y_pred_played = (played_df[predicted_prob_col_gw] > 0.5).astype(int)
                played_accuracies.append(accuracy_score(y_true_played, y_pred_played))

            axes[1].plot(fpr, tpr, lw=2.5, label=f'GW+{i} (AUC = {roc_auc:.2f})')

        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate', fontsize=10)
        axes[1].set_ylabel('True Positive Rate', fontsize=10)
        axes[1].set_title('ROC Curves by Gameweek Horizon', fontsize=14)
        axes[1].legend(loc="lower right")
        axes[1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout(pad=3.0)
        plt.savefig('performance_summary_plot.png', dpi=300)
        plt.close()
        print("Generated 'performance_summary_plot.png'")

        # --- Prediction Error Density Plot (remains separate) ---
        plt.figure(figsize=(12, 8))
        maes = []
        point_accuracies = []
        point_accuracies_gt_zero = []
        spearman_correlations = []
        palette = sns.color_palette("viridis", 6)

        for i in range(1, 7):
            pred_col = f'predicted_points_gw+{i}'
            actual_col = f'points_gw+{i}'
            if pred_col in df.columns and actual_col in df.columns:
                temp_df = df[[pred_col, actual_col]].dropna()
                if temp_df.shape[0] > 1:  # Spearman needs at least 2 data points
                    error = temp_df[pred_col] - temp_df[actual_col]
                    sns.kdeplot(error, label=f'GW+{i}', fill=True, alpha=0.1, lw=2.5, color=palette[i - 1])
                    maes.append(mean_absolute_error(temp_df[actual_col], temp_df[pred_col]))
                    point_accuracies.append(np.mean(np.abs(error) <= 1))

                    # Calculate Spearman's Rank Correlation
                    corr, _ = spearmanr(temp_df[actual_col], temp_df[pred_col])
                    spearman_correlations.append(corr)

                    scored_df = temp_df[temp_df[actual_col] > 0]
                    if not scored_df.empty:
                        error_gt_zero = scored_df[pred_col] - scored_df[actual_col]
                        point_accuracies_gt_zero.append(np.mean(np.abs(error_gt_zero) <= 1))

        plt.axvline(0, color='black', linestyle='--', lw=1.5)
        plt.title('Density of Prediction Errors by Gameweek Horizon', fontsize=16)
        plt.xlabel('Prediction Error (Predicted - Actual)', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.legend()
        plt.xlim(-10, 10)
        plt.savefig('error_density_by_gameweek.png')
        plt.close()
        print("Generated 'error_density_by_gameweek.png'")

        # --- Calculate and Print Metrics for Slide ---
        gw1_mae_df = df[[actual_points_col, predicted_points_col]].dropna()
        gw1_mae = mean_absolute_error(gw1_mae_df[actual_points_col], gw1_mae_df[predicted_points_col])

        print("\n--- Performance Metrics for Presentation Slide ---")
        if spearman_correlations:
            print(f"Average Spearman's Rank Correlation: {np.mean(spearman_correlations):.4f}")
        if all_class_accuracies:
            print(f"Average Minutes Prediction Accuracy (All Players): {np.mean(all_class_accuracies):.2%}")
        if played_accuracies:
            print(f"Average Minutes Prediction Accuracy (Played > 0 Mins): {np.mean(played_accuracies):.2%}")
        if point_accuracies:
            print(f"Average Points Accuracy (All Players, +/- 1 Point): {np.mean(point_accuracies):.2%}")
        if point_accuracies_gt_zero:
            print(f"Average Points Accuracy (Scored > 0 Pts, +/- 1 Point): {np.mean(point_accuracies_gt_zero):.2%}")
        if all_roc_auc:
            print(f"Average Minutes Prediction AUC: {np.mean(all_roc_auc):.4f}")
        if maes:
            print(f"Average Points Prediction MAE: {np.mean(maes):.4f}")
        print(f"GW+1 Points Prediction MAE: {gw1_mae:.4f}")

    if __name__ == '__main__':
        data_file_path = '../../data/output/predictions/performance_data.csv'
        plot_performance_visuals(data_file_path)


if __name__ == '__main__':
    # Set the path to your data file
    data_file_path = '../../data/output/predictions/performance_data.csv'
    plot_performance_visuals(data_file_path)
