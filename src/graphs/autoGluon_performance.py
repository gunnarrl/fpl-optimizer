import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV (update the filename if needed)
df = pd.read_csv('../../data/processed/model_performance_autogluon_parallel.csv')

# Ensure all MAE scores are positive
df['abs_mae'] = df['score_val'].abs()
best_mae = df['abs_mae'].mean()
best_model = df.loc[df['abs_mae'].idxmin(), 'model']
avg_pred_time = df['pred_time_val'].mean()
avg_fit_time = df['fit_time'].mean()
used_stacking = df['stack_level'].max() > 0
best_models = df.loc[df.groupby(['horizon', 'GW'])['abs_mae'].idxmin()]

# Count how often each model was best
model_counts = best_models['model'].value_counts()

# Convert to percentage
model_percentages = (model_counts / model_counts.sum() * 100).round(2)

# Combine into one table
model_summary = pd.DataFrame({
    'Times Best': model_counts,
    'Percentage': model_percentages
}).sort_values('Times Best', ascending=False)

print("🔢 Percentage of Times Each Model Was Best:")
print(model_summary)
# Print summary
print("🔹 AutoGluon Model Performance Summary 🔹")
print(f"Best MAE: {best_mae:.3f}")
print(f"Best Model: {best_model}")
print(f"Average Fit Time: {avg_fit_time:.2f} sec")
print(f"Average Prediction Time: {avg_pred_time:.4f} sec")
print(f"Used Stacking: {'Yes' if used_stacking else 'No'}")
print(f"Average MAE: {df['abs_mae'].mean():.3f}")
print(f"Median MAE: {df['abs_mae'].median():.3f}")
print(f"MAE Std Dev: {df['abs_mae'].std():.3f}")
fastest_model = df.loc[df['pred_time_val'].idxmin(), 'model']
slowest_model = df.loc[df['pred_time_val'].idxmax(), 'model']
print(f"Fastest Predictor: {fastest_model}")
print(f"Slowest Predictor: {slowest_model}")

print("Model count per stack level:")
print(df['stack_level'].value_counts().sort_index())
print()

best_by_level = df.groupby('stack_level')['abs_mae'].min()
print("Best MAE by stack level:")
print(best_by_level)

print("Most frequent best model:")
print(df.loc[df.groupby(['horizon', 'GW'])['abs_mae'].idxmin(), 'model'].value_counts())

print()
# 1. Summary Statistics by Model (using abs_mae)
summary = df.groupby('model')[['abs_mae', 'fit_time', 'pred_time_val']].agg(['mean', 'median', 'std'])
summary.columns = ['_'.join(col) for col in summary.columns]
summary = summary.sort_values('abs_mae_mean')
summary.reset_index(inplace=True)

# 2. Best Model per Horizon and Game Week
best = df.loc[df.groupby(['horizon', 'GW'])['abs_mae'].idxmin()].sort_values(['horizon', 'GW'])

# 3. Bar Plot: Mean Absolute MAE by Model
mean_mae_by_model = df.groupby('model')['abs_mae'].mean().sort_values()
plt.figure(figsize=(12, 6))
plt.bar(mean_mae_by_model.index, mean_mae_by_model.values, color='skyblue')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Mean Absolute MAE')
plt.xlabel('Model')
plt.title('Mean Absolute MAE per Model')
plt.tight_layout()
plt.show()

# 4. Scatter Plot: MAE vs. Fit Time
plt.figure(figsize=(8, 6))
plt.scatter(df['fit_time'], df['abs_mae'], alpha=0.7)
plt.xlabel('Fit Time (s)')
plt.ylabel('Absolute MAE')
plt.title('Model Absolute MAE vs Fit Time')
plt.tight_layout()
plt.show()

# 5. Scatter Plot: MAE vs. Prediction Time
plt.figure(figsize=(8, 6))
plt.scatter(df['pred_time_val'], df['abs_mae'], alpha=0.7, color='orange')
plt.xlabel('Prediction Time (s)')
plt.ylabel('Absolute MAE')
plt.title('Model Absolute MAE vs Prediction Time')
plt.tight_layout()
plt.show()

# 6. Boxplot: MAE Distribution by Stack Level
df['stack_level'] = df['stack_level'].astype(int)
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='stack_level', y='abs_mae', palette="Set2")
plt.xlabel('Stack Level')
plt.ylabel('Absolute MAE')
plt.title('Absolute MAE Distribution by Stack Level')
plt.tight_layout()
plt.show()

# 7. Line Plot: Best MAE by Horizon
best_horizon = df.groupby('horizon')['abs_mae'].min().reset_index()
plt.figure(figsize=(8, 6))
plt.plot(best_horizon['horizon'], best_horizon['abs_mae'], marker='o')
plt.xlabel('Horizon')
plt.ylabel('Best Absolute MAE')
plt.title('Best Absolute MAE by Horizon')
plt.tight_layout()
plt.show()

# 8. Heatmap: Best MAE per Game Week (GW) and Horizon
pivot = df.pivot_table(index='GW', columns='horizon', values='abs_mae', aggfunc='min')
plt.figure(figsize=(12, 8))
sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis")
plt.title('Heatmap of Best Absolute MAE by GW and Horizon')
plt.tight_layout()
plt.savefig("heatmap")
plt.show()
