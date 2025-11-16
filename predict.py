import os
from typing import Dict, Any

import numpy as np
import pandas as pd

from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")


def _load_and_engineer_features(csv_path: str = "sales_data.csv"):
    """
    Load the raw CSV and create two feature sets:

    * baseline_data: no lag/rolling features, just original columns + date parts
    * full_data: includes lag and rolling statistics

    Returns
    -------
    (baseline_data, full_data, log_text)
    """
    log_lines = []

    def log(msg: str):
        log_lines.append(msg)

    log(f"Loading data from {csv_path}")
    data = pd.read_csv(csv_path)

    # Basic cleaning
    leakage_cols = ["Demand", "Units Ordered"]
    data = data.drop(columns=[c for c in leakage_cols if c in data.columns])

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)

    # Date features
    data["DayOfWeek"] = data["Date"].dt.dayofweek
    data["Month"] = data["Date"].dt.month
    data["Year"] = data["Date"].dt.year
    data["DaysSinceStart"] = (data["Date"] - data["Date"].min()).dt.days

    # Baseline copy before lag/rolling
    baseline_data = data.copy()

    # Lag features
    log("Creating lag and rolling features...")
    data["Lag_1"] = data.groupby(["Store ID", "Product ID"])["Units Sold"].shift(1)
    data["Lag_7"] = data.groupby(["Store ID", "Product ID"])["Units Sold"].shift(7)
    data["Lag_14"] = data.groupby(["Store ID", "Product ID"])["Units Sold"].shift(14)
    data["Lag_30"] = data.groupby(["Store ID", "Product ID"])["Units Sold"].shift(30)

    # Rolling stats on Units Sold
    for window in [7, 14, 30]:
        data[f"Rolling_{window}_mean"] = (
            data.groupby(["Store ID", "Product ID"])["Units Sold"]
            .shift(1)
            .rolling(window, min_periods=1)
            .mean()
        )
        data[f"Rolling_{window}_std"] = (
            data.groupby(["Store ID", "Product ID"])["Units Sold"]
            .shift(1)
            .rolling(window, min_periods=1)
            .std()
        )

    # Rolling price features
    if "Price" in data.columns:
        data["Rolling_Price_7"] = (
            data.groupby(["Store ID", "Product ID"])["Price"]
            .shift(1)
            .rolling(7, min_periods=1)
            .mean()
        )
        data["PriceChange_7"] = data["Price"] - data["Rolling_Price_7"]

    # Rolling inventory features
    if "Inventory Level" in data.columns:
        data["Rolling_Inventory_7"] = (
            data.groupby(["Store ID", "Product ID"])["Inventory Level"]
            .shift(1)
            .rolling(7, min_periods=1)
            .mean()
        )
        data["InventoryChange_7"] = (
            data["Inventory Level"] - data["Rolling_Inventory_7"]
        )

    # Fill NaNs on rolling/lag features
    for col in data.columns:
        if "Rolling" in col or "Lag" in col:
            data[col] = data.groupby(["Store ID", "Product ID"])[col].ffill()
            data[col] = data[col].fillna(0)

    # Drop rows where first lag is still missing
    data = data.dropna(subset=["Lag_1"])

    # Fill any remaining numeric NaNs
    num_cols = data.select_dtypes(include=[np.number]).columns
    data[num_cols] = data[num_cols].fillna(0)

    log(
        f"Baseline data shape: {baseline_data.shape}, "
        f"Full data shape after lags: {data.shape}"
    )

    return baseline_data, data, "\n".join(log_lines)


def train_and_evaluate_models(
    csv_path: str = "sales_data.csv",
    plot_path="static/model_accuracy_comparison.png",
) -> Dict[str, Any]:
    """
    Train baseline and feature-rich models and return evaluation metrics.

    Designed for use from a /train endpoint. It does *not* save the model
    used by the prediction API – that is handled by
    predict_sales.train_and_save_model.

    Returns
    -------
    dict
        {
          "model_name": best model by test RMSE,
          "train_rmse": train RMSE of best model,
          "test_rmse": test RMSE of best model,
          "metrics": { model_name -> mean test RMSE },
          "log": multi-line training summary,
          "plot_path": path to saved PNG
        }
    """
    base_data, full_data, prep_log = _load_and_engineer_features(csv_path)

    logs = [prep_log]

    # ---------- Baseline model (no lag/rolling) ----------
    y_base = base_data["Units Sold"]
    X_base = base_data.drop(columns=["Units Sold", "Date"])

    cat_base = X_base.select_dtypes(include=["object"]).columns.tolist()
    num_base = X_base.select_dtypes(include=[np.number]).columns.tolist()

    preproc_base = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
                cat_base,
            ),
            ("num", StandardScaler(), num_base),
        ],
        remainder="drop",
    )

    baseline_pipeline = Pipeline(
        steps=[
            ("preprocessor", preproc_base),
            ("model", LinearRegression()),
        ]
    )

    tscv = TimeSeriesSplit(n_splits=3)
    logs.append("Running baseline LinearRegression (no lag/rolling) ...")
    res_base = cross_validate(
        baseline_pipeline,
        X_base,
        y_base,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        return_train_score=True,
    )
    baseline_train_rmse = -res_base["train_score"]
    baseline_test_rmse = -res_base["test_score"]

    # ---------- Models with lag/rolling ----------
    y = full_data["Units Sold"]
    X = full_data.drop(columns=["Units Sold", "Date"])

    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    preproc_full = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
                cat_cols,
            ),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop",
    )

    # Linear regression with lags
    logs.append("Running LinearRegression with lag/rolling features ...")
    lin_pipeline = Pipeline(
        steps=[("preprocessor", preproc_full), ("model", LinearRegression())]
    )
    res_lin = cross_validate(
        lin_pipeline,
        X,
        y,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        return_train_score=True,
    )
    lin_train_rmse = -res_lin["train_score"]
    lin_test_rmse = -res_lin["test_score"]

    # Decision tree with lags
    logs.append("Running DecisionTreeRegressor with lag/rolling features ...")
    dt_pipeline = Pipeline(
        steps=[
            ("preprocessor", preproc_full),
            (
                "model",
                DecisionTreeRegressor(
                    max_depth=15,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42,
                ),
            ),
        ]
    )
    res_dt = cross_validate(
        dt_pipeline,
        X,
        y,
        cv=tscv,
        scoring="neg_root_mean_squared_error",
        return_train_score=True,
    )
    dt_train_rmse = -res_dt["train_score"]
    dt_test_rmse = -res_dt["test_score"]

    # ---------- Aggregate metrics ----------
    baseline_mean = float(baseline_test_rmse.mean())
    lin_mean = float(lin_test_rmse.mean())
    dt_mean = float(dt_test_rmse.mean())

    metrics = {
        "Linear Regression (Baseline)": baseline_mean,
        "Linear Regression (Lag/Rolling)": lin_mean,
        "Decision Tree (Lag/Rolling)": dt_mean,
    }

    best_model_name = min(metrics.items(), key=lambda kv: kv[1])[0]
    if best_model_name == "Linear Regression (Baseline)":
        best_train = float(baseline_train_rmse.mean())
        best_test = baseline_mean
    elif best_model_name == "Linear Regression (Lag/Rolling)":
        best_train = float(lin_train_rmse.mean())
        best_test = lin_mean
    else:
        best_train = float(dt_train_rmse.mean())
        best_test = dt_mean

    # ---------- Build log text ----------
    logs.append("")
    logs.append("Cross-validated RMSE (mean over folds):")
    logs.append(
        f"  Baseline LinearRegression:   "
        f"train={baseline_train_rmse.mean():.3f}, test={baseline_mean:.3f}"
    )
    logs.append(
        f"  LinearRegression + lags:     "
        f"train={lin_train_rmse.mean():.3f}, test={lin_mean:.3f}"
    )
    logs.append(
        f"  DecisionTree + lags:         "
        f"train={dt_train_rmse.mean():.3f}, test={dt_mean:.3f}"
    )
    logs.append("")
    logs.append(f"Best model by test RMSE: {best_model_name} (RMSE={best_test:.3f})")

    full_log = "\n".join(logs)

    # ---------- Visualization ----------
    model_names = list(metrics.keys())
    rmse_values = [metrics[m] for m in model_names]

    plot_dir = os.path.dirname(plot_path)
    if plot_dir and not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(model_names, rmse_values)
    ax.set_ylabel("Test RMSE")
    ax.set_title("Model Test RMSE Comparison")
    ax.tick_params(axis="x", rotation=20)

    for bar, val in zip(bars, rmse_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

    return {
        "model_name": best_model_name,
        "train_rmse": best_train,
        "test_rmse": best_test,
        "metrics": metrics,
        "log": full_log,
        "plot_path": plot_path,
    }
