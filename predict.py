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
# PREPARE TARGET AND FEATURES
# ============================================================

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
print("BASELINE LINEAR REGRESSION MODEL")
print("="*70)

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

