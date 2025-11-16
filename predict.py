import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


def train_inventory_model(csv_path: str = "retail_store_inventory.csv", n_splits: int = 10):
    data = pd.read_csv(csv_path)

    data = data.drop(columns=["Demand Forecast", "Units Ordered"])

    data["Date"] = pd.to_datetime(data["Date"])
    data["Date"] = (data["Date"] - data["Date"].min()).dt.days

    y = data["Units Sold"]
    X = data.drop(columns=["Units Sold"])

    # Categorical + numeric columns
    categorical_cols = [
        "Store ID",
        "Product ID",
        "Category",
        "Region",
        "Weather Condition",
        "Seasonality",
        "Holiday/Promotion",
    ]

    numeric_cols = [
        "Date",
        "Inventory Level",
        "Price",
        "Discount",
        "Competitor Pricing",
    ]

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(), categorical_cols),
            ("numeric", StandardScaler(), numeric_cols),
        ]
    )

    model = LinearRegression()

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    # Time series cross-validation
    cross_val = TimeSeriesSplit(n_splits=n_splits)

    # Evaluate with RMSE
    results = cross_validate(
        pipeline,
        X,
        y,
        cv=cross_val,
        scoring="neg_root_mean_squared_error",
        return_estimator=True,
    )

    # Fitted models from each fold
    models = results["estimator"]

    # Feature names from the preprocessor
    features = models[0].named_steps["preprocessor"].get_feature_names_out()

    # Stack coefficients across folds
    coef_matrix = np.vstack([m.named_steps["model"].coef_ for m in models])

    # Average and abs
    avg_coefs = coef_matrix.mean(axis=0)

    coef_summary = (
        pd.DataFrame(
            {
                "Feature": features,
                "Mean Coefficient": avg_coefs,
                "Abs(Mean)": np.abs(avg_coefs),
            }
        )
        .sort_values(by="Abs(Mean)", ascending=False)
        .reset_index(drop=True)
    )

    # Convert sklearn’s negative RMSE to positive RMSE
    rmse_scores = -results["test_score"]
    mean_rmse = rmse_scores.mean()

    metrics = {
        "rmse_scores": rmse_scores,
        "mean_rmse": mean_rmse,
    }

    return metrics, coef_summary


if __name__ == "__main__":
    metrics, coef_summary = train_inventory_model()
    print(coef_summary.head(10))
    print("RMSE for each fold: ", metrics["rmse_scores"])
    print("Mean RMSE: ", metrics["mean_rmse"])