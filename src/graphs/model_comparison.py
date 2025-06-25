import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    median_absolute_error
)
from scipy.stats import spearmanr, ttest_rel, wilcoxon

# === CONFIG ===
file1 = '../../data/output/predictions/koa_predictions_updated.csv'
file2 = '../../data/output/predictions/predictions_autogluon_parallel.csv'
model1_name = 'xGBoost'
model2_name = 'AutoGluon'
top_k = 10
horizons = range(1, 7)

# === LOAD & LABEL ===
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
df1['model'] = model1_name
df2['model'] = model2_name
df = pd.concat([df1, df2], ignore_index=True)

# === METRICS BY HORIZON ===
records = []
for model in df['model'].unique():
    sub = df[df['model'] == model]
    for h in horizons:
        y_true = sub[f'points_gw+{h}']
        y_pred = sub[f'predicted_points_gw+{h}']
        mask = y_true.notna() & y_pred.notna()
        yt, yp = y_true[mask], y_pred[mask]

        mae  = mean_absolute_error(yt, yp)
        rmse = np.sqrt(mean_squared_error(yt, yp))
        r2   = r2_score(yt, yp)
        mape = (np.abs(yt - yp) / yt).replace([np.inf, -np.inf], np.nan).dropna().mean() * 100
        medae = median_absolute_error(yt, yp)
        rho, _ = spearmanr(yt, yp)

        records.append({
            'model': model,
            'horizon': h,
            'MAE': mae,
            'RMSE': rmse,
            'R2': r2,
            'MAPE': mape,
            'MedAE': medae,
            'Spearman': rho
        })

metrics_df = pd.DataFrame(records)

# === SUMMARY METRICS ===
summary = metrics_df.groupby('model').agg(
    avg_MAE=('MAE', 'mean'),
    avg_MAPE=('MAPE', 'mean'),
    avg_R2=('R2', 'mean'),
    avg_spearman=('Spearman', 'mean')
).reset_index()

# === Residual Variance ===
resid_stats = []
for model in df['model'].unique():
    all_resid = []
    for h in horizons:
        yt = df.loc[df['model'] == model, f'points_gw+{h}']
        yp = df.loc[df['model'] == model, f'predicted_points_gw+{h}']
        mask = yt.notna() & yp.notna()
        res = (yp[mask] - yt[mask])
        all_resid.append(res)
    combined_resid = pd.concat(all_resid)
    resid_stats.append({'model': model, 'resid_variance': np.var(combined_resid)})

resid_summary = pd.DataFrame(resid_stats)
summary = summary.merge(resid_summary, on='model')

# === MAE % IMPROVEMENT ===
m1 = summary.loc[summary['model']==model1_name,'avg_MAE'].values[0]
m2 = summary.loc[summary['model']==model2_name,'avg_MAE'].values[0]
pct_imp = (m2 - m1) / m2 * 100

# === TOP-10 HIT RATE & CUMULATIVE POINTS (GW+1) ===
hit_rates = []
cum_pts = []
for model in df['model'].unique():
    sub = df[df['model']==model]
    grouped = sub.groupby('GW')
    hits = []
    cum = []
    for gw, g in grouped:
        valid = g.dropna(subset=['predicted_points_gw+1','points_gw+1'])
        pred_top = valid.nlargest(top_k, 'predicted_points_gw+1')['element']
        actual_top = valid.nlargest(top_k, 'points_gw+1')['element']
        hits.append(len(set(pred_top) & set(actual_top)) / top_k)
        cum.append(valid.set_index('element').loc[pred_top,'points_gw+1'].sum())
    hit_rates.append({'model':model, 'hit_rate':np.mean(hits)})
    perfect = [
        df[(df['model']==model)&(df['GW']==gw)]
         .nlargest(top_k,'points_gw+1')['points_gw+1'].sum()
        for gw in df['GW'].unique()
    ]
    cum_pts.append({'model':model, 'cum_pts_pct': np.mean(cum)/np.mean(perfect)*100 })

hit_df = pd.DataFrame(hit_rates).merge(pd.DataFrame(cum_pts), on='model')

# === STATISTICAL SIGNIFICANCE TEST (GW+1 MAE) ===
true_vals = df1['points_gw+1'].dropna()
pred1 = df1.loc[true_vals.index, 'predicted_points_gw+1']
pred2 = df2.loc[true_vals.index, 'predicted_points_gw+1']
errors1 = (pred1 - true_vals).abs()
errors2 = (pred2 - true_vals).abs()
mask = errors1.notna() & errors2.notna()
e1 = errors1[mask]
e2 = errors2[mask]
t_stat, p_value = ttest_rel(e1, e2)
_, w_p = wilcoxon(e1, e2)

# === PRINT SUMMARY ===
print("\n=== Model Comparison Summary ===")
print(summary[['model', 'avg_MAE', 'avg_MAPE', 'avg_R2', 'avg_spearman', 'resid_variance']])
print(f"\n{model1_name} improves MAE by {pct_imp:.1f}% vs {model2_name}")
print("\nTop-10 hit rate and % of perfect points (GW+1):")
print(hit_df.to_string(index=False))
print(f"\nStatistical Significance Tests (GW+1 MAE):")
print(f"Paired t-test p-value: {p_value:.4f}")
print(f"Wilcoxon test p-value: {w_p:.4f}")

# === PLOTS ===
for metric in ['MAE','RMSE','R2','MAPE']:
    plt.figure(figsize=(8,5))
    pivot = metrics_df.pivot(index='horizon', columns='model', values=metric)
    pivot.plot.bar(ax=plt.gca(), width=0.8)
    plt.title(f"{metric} by Forecast Horizon")
    plt.xlabel("Forecast Horizon (GW ahead)")
    plt.ylabel(metric + (" (%)" if metric=='MAPE' else ""))
    plt.tight_layout()
    if metric =='MAE':
        plt.savefig('gw_MAE')
    plt.show()

# === BOX PLOT: ABSOLUTE ERROR ===
df_err = []
for model in df['model'].unique():
    sub = df[df['model']==model]
    for h in horizons:
        err = (sub[f'points_gw+{h}'] - sub[f'predicted_points_gw+{h}']).abs().dropna()
        df_err.append(pd.DataFrame({'model': model, 'abs_err': err}))
err_df = pd.concat(df_err, ignore_index=True)
plt.figure(figsize=(6,5))
err_df.boxplot(by='model', column='abs_err')
plt.title("Absolute Error Distribution")
plt.suptitle("")
plt.ylabel("Absolute Error")
plt.show()

# === RESIDUAL HISTOGRAM ===
residuals = []
for model in df['model'].unique():
    model_resids = []
    for h in horizons:
        actual = df.loc[df['model'] == model, f'points_gw+{h}']
        pred   = df.loc[df['model'] == model, f'predicted_points_gw+{h}']
        r = (pred - actual).dropna()
        model_resids.append(r)
    full_resid = pd.concat(model_resids)
    residuals.append(pd.DataFrame({'residual': full_resid, 'model': model}))
residuals_df = pd.concat(residuals)
plt.figure(figsize=(7, 5))
for model in residuals_df['model'].unique():
    subset = residuals_df[residuals_df['model'] == model]['residual']
    plt.hist(subset, bins=30, alpha=0.6, label=model)
plt.title("Residuals Histogram (Prediction – Actual)")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()