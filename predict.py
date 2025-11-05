import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline


# Load the data
data = pd.read_csv("retail_store_inventory.csv")

# Dropping demand forecast, cause its cheating if we have that. Also dropping Units Ordered because it doesn't affect future sales.
data = data.drop(columns=["Demand Forecast", "Units Ordered"])

# Converting Date to numeric
data["Date"] = pd.to_datetime(data["Date"])
data["Date"] = (data["Date"] - data["Date"].min()).dt.days # this just converts the date into the number of days since the earliest date in the dataset

# Define target and features
y = data["Units Sold"]
X = data.drop(columns = ["Units Sold"]) # features are everything except units sold

# Identify categorical and numerical columns
categorical_cols = [
    "Store ID",
    "Product ID",
    "Category",
    "Region",
    "Weather Condition",
    "Seasonality",
    "Holiday/Promotion"
]

numeric_cols = [
    "Date",
    "Inventory Level",
    "Price",
    "Discount",
    "Competitor Pricing"
]

# This is data preprocessing "transformer"
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(), categorical_cols), # apply one-hot encoding to the categorical columns
        ("numeric", StandardScaler(), numeric_cols) # apply standardization to the numeric columns
    ]
)

# Define the linear regression model
model = LinearRegression()

# Build the pipeline
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor), # Step 1: apply column transformations (one-hot encoding)
    ("model", model) # Step 2: fit the linear regression model (sklearn doesn't use gradient descent and just gives a closed form solution via linear algebra)
])

# User time series cross-validation (using normal k-fold would mess up our chronological data)
cross_val = TimeSeriesSplit(n_splits=10) # For each split, it trains on earlier data and tests on later data.

# Evaluate the model using RMSE
results = cross_validate(
    pipeline, X, y,
    cv=cross_val,
    scoring="neg_root_mean_squared_error",
    return_estimator=True
)

# Access fitted models (one per fold)
models = results["estimator"]

# Use the feature names from one fitted model
features = models[0].named_steps["preprocessor"].get_feature_names_out()

# Extract all coefficient arrays
coef_matrix = np.vstack([
    m.named_steps["model"].coef_ for m in models
])

# Compute average and std deviation across folds
avg_coefs = coef_matrix.mean(axis=0)
std_coefs = coef_matrix.std(axis=0)

# Combine into a DataFrame
coef_summary = pd.DataFrame({
    "Feature": features,
    "Mean Coefficient": avg_coefs,
    "Abs(Mean)": np.abs(avg_coefs)
}).sort_values(by="Abs(Mean)", ascending=False)

print(coef_summary.head(10))

# convert to positive because sklearn only has negative RMSE for some reason
rmse_scores = -results["test_score"]
print("RMSE for each fold: ", rmse_scores)
print("Mean RMSE: ", rmse_scores.mean())