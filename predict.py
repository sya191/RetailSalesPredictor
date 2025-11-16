import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# LOAD + CLEAN DATA
# ============================================================

print("Loading data...")
data = pd.read_csv("sales_data.csv")

print(f"Dataset shape: {data.shape}")
print(f"Columns: {list(data.columns)}")

# Drop leakage columns (Demand likely predicts Units Sold)
leakage_cols = ["Demand", "Units Ordered"]
data = data.drop(columns=[col for col in leakage_cols if col in data.columns])

# Convert date to datetime
data["Date"] = pd.to_datetime(data["Date"])

# Sort by date for time series cross-validation
data = data.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)


# ============================================================
# BASIC FEATURE ENGINEERING
# ============================================================

# Extract date features
data["DayOfWeek"] = data["Date"].dt.dayofweek
data["Month"] = data["Date"].dt.month
data["Year"] = data["Date"].dt.year

# Days since start
data["DaysSinceStart"] = (data["Date"] - data["Date"].min()).dt.days


# ============================================================
# CREATE LAG + ROLLING FEATURES
# ============================================================

print("\nCreating lag and rolling features...")

# Lag features - shift to avoid leakage
data["Lag_1"] = data.groupby(["Store ID", "Product ID"])["Units Sold"].shift(1)
data["Lag_7"] = data.groupby(["Store ID", "Product ID"])["Units Sold"].shift(7)
data["Lag_14"] = data.groupby(["Store ID", "Product ID"])["Units Sold"].shift(14)
data["Lag_30"] = data.groupby(["Store ID", "Product ID"])["Units Sold"].shift(30)

# Rolling statistics (shift to avoid leakage)
for window in [7, 14, 30]:
    data[f"Rolling_{window}_mean"] = (
        data.groupby(["Store ID", "Product ID"])["Units Sold"]
            .shift(1).rolling(window, min_periods=1).mean()
    )
    data[f"Rolling_{window}_std"] = (
        data.groupby(["Store ID", "Product ID"])["Units Sold"]
            .shift(1).rolling(window, min_periods=1).std()
    )

# Rolling price features
data["Rolling_Price_7"] = (
    data.groupby(["Store ID", "Product ID"])["Price"]
        .shift(1).rolling(7, min_periods=1).mean()
)
data["PriceChange_7"] = data["Price"] - data["Rolling_Price_7"]

# Rolling inventory features
data["Rolling_Inventory_7"] = (
    data.groupby(["Store ID", "Product ID"])["Inventory Level"]
        .shift(1).rolling(7, min_periods=1).mean()
)
data["InventoryChange_7"] = data["Inventory Level"] - data["Rolling_Inventory_7"]

print("Lag and rolling features created.")


# ============================================================
# PREPARE TARGET AND FEATURES
# ============================================================

# Fill NaN values in rolling features (they can have NaN at the start)
# Forward fill within each group, then fill remaining with 0
for col in data.columns:
    if 'Rolling' in col or 'Lag' in col:
        # Forward fill within each store-product group
        data[col] = data.groupby(["Store ID", "Product ID"])[col].ffill()
        # Fill any remaining NaN with 0
        data[col] = data[col].fillna(0)

# Drop rows where Lag_1 is still missing (very beginning of time series)
data = data.dropna(subset=["Lag_1"])

# Final check: fill any remaining NaN values in numeric columns
numeric_cols_before = data.select_dtypes(include=[np.number]).columns
data[numeric_cols_before] = data[numeric_cols_before].fillna(0)

y = data["Units Sold"]
X = data.drop(columns=["Units Sold", "Date"])

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numeric columns: {numeric_cols}")
print(f"\nTarget variable (Units Sold) statistics:")
print(f"  Mean: {y.mean():.2f}")
print(f"  Std: {y.std():.2f}")
print(f"  Min: {y.min()}")
print(f"  Max: {y.max()}")


# ============================================================
# PREPROCESSING PIPELINE
# ============================================================

# One-hot encode categoricals, standardize numerics
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols),
        ("num", StandardScaler(), numeric_cols)
    ],
    remainder='drop'
)


# ============================================================
# BASELINE LINEAR REGRESSION MODEL
# ============================================================

print("\n" + "="*70)
print("LINEAR REGRESSION MODEL WITH LAG & ROLLING FEATURES")
print("="*70)
print(f"\nData shape after lag features: {data.shape}")
print(f"Number of lag/rolling features added: {len([col for col in numeric_cols if 'Lag' in col or 'Rolling' in col])}")

# Create pipeline
linear_model = LinearRegression()
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", linear_model)
])

# Time series cross-validation
cross_val = TimeSeriesSplit(n_splits=3)

print("\nPerforming time series cross-validation...")
results = cross_validate(
    pipeline, X, y,
    cv=cross_val,
    scoring="neg_root_mean_squared_error",
    return_estimator=True,
    return_train_score=True
)

# Calculate RMSE
rmse_scores = -results["test_score"]
train_rmse_scores = -results["train_score"]

print(f"\nTrain RMSE for each fold: {train_rmse_scores}")
print(f"Test RMSE for each fold: {rmse_scores}")
print(f"\nMean Train RMSE: {train_rmse_scores.mean():.4f}")
print(f"Mean Test RMSE: {rmse_scores.mean():.4f}")
print(f"Std Test RMSE: {rmse_scores.std():.4f}")


# ============================================================
# FEATURE CORRELATION ANALYSIS
# ============================================================

print("\n" + "="*70)
print("FEATURE CORRELATION WITH TARGET")
print("="*70)

# Check correlation of numeric features with target
numeric_data = data[numeric_cols + ["Units Sold"]]
correlations = numeric_data.corr()["Units Sold"].abs().sort_values(ascending=False)

print("\nTop correlations with Units Sold:")
for feature, corr in correlations.head(10).items():
    if feature != "Units Sold":
        print(f"  {feature}: {corr:.4f}")


# ============================================================
# MODEL COEFFICIENTS (if possible)
# ============================================================

print("\n" + "="*70)
print("MODEL SUMMARY")
print("="*70)

# Fit on full data to get coefficients
pipeline.fit(X, y)
final_model = pipeline.named_steps['model']

print(f"\nR-squared (on full data): {final_model.score(pipeline.named_steps['preprocessor'].transform(X), y):.4f}")
print(f"Intercept: {final_model.intercept_:.4f}")
print(f"Number of features: {len(final_model.coef_)}")

# Get feature names after preprocessing
try:
    # Get feature names from one-hot encoder and numeric columns
    cat_encoder = pipeline.named_steps['preprocessor'].named_transformers_['cat']
    num_features = numeric_cols
    cat_features = []
    
    for i, col in enumerate(categorical_cols):
        categories = cat_encoder.categories_[i]
        for cat in categories[1:]:  # Skip first category (dropped)
            cat_features.append(f"{col}_{cat}")
    
    all_feature_names = cat_features + num_features
    
    # Get top coefficients
    coef_df = pd.DataFrame({
        'Feature': all_feature_names[:len(final_model.coef_)],
        'Coefficient': final_model.coef_[:len(all_feature_names)]
    })
    coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
    
    print(f"\nTop 10 most important features (by coefficient magnitude):")
    for idx, row in coef_df.head(10).iterrows():
        print(f"  {row['Feature']}: {row['Coefficient']:.4f}")
except Exception as e:
    print(f"\nCould not extract feature names: {e}")

print("\n" + "="*70)
print("BASELINE MODEL COMPLETE")
print("="*70)
print(f"Final Test RMSE: {rmse_scores.mean():.4f}")

