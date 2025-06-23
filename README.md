# FPL AI Optimizer

This project provides a complete, end-to-end pipeline for optimizing Fantasy Premier League (FPL) team selections using machine learning and mathematical optimization. The system ingests data from multiple sources, engineers a rich set of features, trains predictive models for player performance, and uses these predictions to recommend the optimal squad for each gameweek, including transfers and chip strategy.

## Features

* **Multi-Source Data Ingestion**: Gathers and merges data from the official FPL API and detailed performance statistics from Understat.
* **Advanced Feature Engineering**: Creates over 100 features, including:
    * Rolling form statistics (3 and 5-gameweek averages).
    * Season-to-date aggregate stats (e.g., `aggregate_xG`, `per_90_minutes`).
    * Team strength and opponent strength metrics.
    * Fixture difficulty and upcoming schedule analysis.
    * Flags for Double Gameweeks (DGWs).
* **Sophisticated Predictive Modeling**:
    * Utilizes a **walk-forward training** approach, retraining models for each gameweek to use the latest available data.
    * Employs a two-stage model: a **Classifier** to predict if a player will play >= 60 minutes and a **Regressor** to predict points.
    * Supports multiple ML frameworks, including **XGBoost**, **Random Forest**, **Neural Networks (PyTorch)**, and **AutoGluon (AutoML)**.
    * Designed for **GPU-accelerated, parallel training** using Dask and `dask-cuda` to handle the intensive computational load.
* **Mathematical Squad Optimization**:
    * Uses the **PuLP** linear programming library to find the mathematically optimal squad.
    * Maximizes predicted points over a **6-week planning horizon**.
    * Adheres to all FPL rules: budget, formations, team limits, etc.
    * Provides optimal transfer recommendations (buy/sell).
    * Includes logic for **strategic chip usage** (Wildcard, Bench Boost, Triple Captain, Free Hit), prioritizing them in high-value gameweeks like DGWs.
* **Evaluation & Visualization**: Includes scripts to generate plots for model performance analysis, comparing predicted points vs. actual points and calculating metrics like Mean Absolute Error (MAE) and R-squared.

## Project Workflow

The project follows a sequential pipeline, where the output of one stage becomes the input for the next.

```
+-------------------------+
|     Data Ingestion      |
| (FPL API, Understat)    |
+-------------------------+
           |
           v
+-------------------------+
|  Feature Engineering    |
| (Form, Aggregates, DGW) |
+-------------------------+
           |
           v
+-------------------------+
|   Modeling & Prediction |
| (XGBoost, AutoML, etc.) |
+-------------------------+
           |
           v
+-------------------------+
| Optimization & Strategy |
| (PuLP, Chip Logic)      |
+-------------------------+
           |
           v
+-------------------------+
|  Team Selection & JSON  |
|      Output Files       |
+-------------------------+
```

## Project Structure

The codebase is organized into modules based on functionality:

```
.
├── data/
│   ├── raw/          # Raw data from sources
│   ├── processed/    # Cleaned and feature-engineered data
│   └── output/       # Final model predictions and team selections
└── src/
    ├── data_ingestion/     # Scripts for fetching and merging data
    ├── feature_engineering/  # Scripts for creating features
    ├── modeling/           # Scripts for training predictive models
    ├── optimization/       # Scripts for squad optimization
    ├── graphs/             # Scripts for performance visualization
    └── utils/              # Helper and utility scripts
```

## How to Run the Pipeline

### 1. Prerequisites

* Python 3.8+
* `pip` for package installation

### 2. Installation

Clone the repository and install the required dependencies. It is highly recommended to use a virtual environment.

```bash
git clone <your-repository-url>
cd <repository-name>
pip install pandas numpy scikit-learn xgboost torch pulp dask dask-cuda autogluon
```

### 3. Execution Order

The scripts are designed to be run in a specific order to ensure the data processing pipeline flows correctly.

**Step 1: Data Ingestion & Feature Engineering**
Run the scripts in the `src/data_ingestion` and `src/feature_engineering` directories. The final output of this stage should be `data/processed/processed_fpl_data.csv`.

*Example Order:*
1.  `src/data_ingestion/add_understat_data.py`
2.  `src/data_ingestion/merge_understat_fpl_data.py`
3.  `src/data_ingestion/add_expected_points.py`
4.  `src/feature_engineering/aggregate_stats.py`
5.  `src/feature_engineering/form_stats.py`
6.  `src/feature_engineering/team_data.py`
7.  `src/feature_engineering/reorder_and_add_labels.py` -> **Creates the final modeling dataset.**

**Step 2: Modeling and Prediction**
Choose a model from the `src/modeling` directory and run it. These scripts will use the `processed_fpl_data.csv` to train models and generate a predictions file (e.g., `data/output/predictions/koa_predictions.csv`).

*Example (using the parallel XGBoost script):*
```bash
python src/modeling/train_predict_parallel.py
```

After generating predictions, you can use the utility script to ensure player metadata is up-to-date:

```bash
python src/utils/fix_pred_file.py
```

**Step 3: Squad Optimization**
Run the optimization script to generate your team for each gameweek. This script will read the predictions file and output a `team_gw<N>.json` file for each week.

*Example (to run a full season optimization):*
```bash
python src/optimization/fpl-optimization.py
```

**Step 4: Evaluation**
Use the scripts in `src/graphs` and `src/utils` to evaluate the performance of your models and the overall season strategy.

*Example:*
```bash
python src/graphs/pred_points_vs_actual.py
python src/utils/season_summary.py
```

## Configuration

Key parameters for the models and optimizer can be configured at the top of their respective Python files. This includes:

* **`PLANNING_HORIZON`**: The number of future gameweeks to consider in predictions and optimization (default: 6).
* **`N_ITER_SEARCH` / `CV_FOLDS`**: Parameters for hyperparameter tuning in `sklearn`.
* **`TIME_LIMIT_PER_MODEL` / `PRESET_QUALITY`**: Configuration for AutoGluon.
* **File Paths**: All input and output file paths are defined and can be adjusted.
* **Optimization Constraints**: Budget, transfer penalties, and chip availability are all defined in `fpl-optimization.py`.
