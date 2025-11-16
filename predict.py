import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")


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
# SAVE BASELINE DATA (before lag/rolling features)
# ============================================================

# Create a copy of data before adding lag/rolling features for baseline model
data_baseline = data.copy()


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
# BASELINE LINEAR REGRESSION (NO LAG/ROLLING FEATURES)
# ============================================================

print("\n" + "="*70)
print("BASELINE LINEAR REGRESSION (NO LAG/ROLLING FEATURES)")
print("="*70)

# Prepare baseline data
y_baseline = data_baseline["Units Sold"]
X_baseline = data_baseline.drop(columns=["Units Sold", "Date"])

# Identify categorical and numeric columns for baseline
categorical_cols_baseline = X_baseline.select_dtypes(include=['object']).columns.tolist()
numeric_cols_baseline = X_baseline.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nBaseline data shape: {data_baseline.shape}")
print(f"Baseline features: {len(numeric_cols_baseline)} numeric, {len(categorical_cols_baseline)} categorical")

# Preprocessor for baseline
preprocessor_baseline = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols_baseline),
        ("num", StandardScaler(), numeric_cols_baseline)
    ],
    remainder='drop'
)

# Baseline model
linear_model_baseline = LinearRegression()
pipeline_baseline = Pipeline(steps=[
    ("preprocessor", preprocessor_baseline),
    ("model", linear_model_baseline)
])

# Time series cross-validation for baseline
cross_val = TimeSeriesSplit(n_splits=3)

print("\nTraining baseline linear regression...")
results_baseline = cross_validate(
    pipeline_baseline, X_baseline, y_baseline,
    cv=cross_val,
    scoring="neg_root_mean_squared_error",
    return_estimator=True,
    return_train_score=True
)

baseline_rmse = -results_baseline["test_score"]
baseline_train_rmse = -results_baseline["train_score"]

print(f"Train RMSE: {baseline_train_rmse.mean():.4f} (folds: {baseline_train_rmse})")
print(f"Test RMSE: {baseline_rmse.mean():.4f} (folds: {baseline_rmse})")


# ============================================================
# PREPROCESSING PIPELINE (for models with lag/rolling)
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
# LINEAR REGRESSION WITH LAG & ROLLING FEATURES
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
print("LINEAR REGRESSION RESULTS")
print("="*70)
print(f"Final Test RMSE: {rmse_scores.mean():.4f}")
linear_rmse = rmse_scores.mean()


# ============================================================
# DECISION TREE MODEL
# ============================================================

print("\n" + "="*70)
print("DECISION TREE REGRESSOR")
print("="*70)

dt_model = DecisionTreeRegressor(
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

dt_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", dt_model)
])

print("Training Decision Tree...")
dt_results = cross_validate(
    dt_pipeline, X, y,
    cv=cross_val,
    scoring="neg_root_mean_squared_error",
    return_estimator=True,
    return_train_score=True
)

dt_rmse = -dt_results["test_score"]
dt_train_rmse = -dt_results["train_score"]

print(f"\nTrain RMSE: {dt_train_rmse.mean():.4f} (folds: {dt_train_rmse})")
print(f"Test RMSE: {dt_rmse.mean():.4f} (folds: {dt_rmse})")


# ============================================================
# MODEL COMPARISON
# ============================================================

print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)

results_comparison = {
    "Linear Regression (Baseline)": baseline_rmse.mean(),
    "Linear Regression (with Lag/Rolling)": linear_rmse,
    "Decision Tree (with Lag/Rolling)": dt_rmse.mean()
}

sorted_models = sorted(results_comparison.items(), key=lambda x: x[1])

print("\nRanking (best to worst RMSE):")
print("-" * 70)
for i, (model_name, rmse_val) in enumerate(sorted_models, 1):
    improvement = linear_rmse - rmse_val
    marker = "***" if i == 1 else "   "
    print(f"{marker} {i}. {model_name:30s} RMSE: {rmse_val:.4f}  (improvement: {improvement:+.4f})")

best_model_name = sorted_models[0][0]
best_rmse = sorted_models[0][1]

print("\n" + "="*70)
print(f"*** BEST MODEL: {best_model_name} with RMSE = {best_rmse:.4f} ***")
print("="*70)

# Calculate improvement percentage
improvement_pct = ((linear_rmse - best_rmse) / linear_rmse) * 100
print(f"\nImprovement over Linear Regression: {improvement_pct:.2f}%")

# Show overfitting analysis
print("\nOverfitting Analysis (Train vs Test RMSE):")
print("-" * 70)
for model_name, train_rmse_val, test_rmse_val in [
    ("Linear Regression (Baseline)", baseline_train_rmse.mean(), baseline_rmse.mean()),
    ("Linear Regression (Lag/Rolling)", train_rmse_scores.mean(), linear_rmse),
    ("Decision Tree (Lag/Rolling)", dt_train_rmse.mean(), dt_rmse.mean())
]:
    gap = train_rmse_val - test_rmse_val
    gap_pct = (gap / test_rmse_val) * 100 if test_rmse_val > 0 else 0
    print(f"{model_name:40s} Train: {train_rmse_val:.4f}  Test: {test_rmse_val:.4f}  Gap: {gap:+.4f} ({gap_pct:+.1f}%)")

# Show improvement over baseline
print("\nImprovement over Baseline Linear Regression:")
print("-" * 70)
baseline_rmse_val = baseline_rmse.mean()
for model_name, rmse_val in sorted_models:
    if model_name != "Linear Regression (Baseline)":
        improvement = baseline_rmse_val - rmse_val
        improvement_pct = (improvement / baseline_rmse_val) * 100
        print(f"{model_name:40s} Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")

print("\n" + "="*70)
print("COMPARISON COMPLETE")
print("="*70)


# ============================================================
# VISUALIZATION: MODEL ACCURACY PLOTS
# ============================================================

print("\n" + "="*70)
print("GENERATING ACCURACY PLOTS")
print("="*70)

# Prepare data for plotting
model_names = list(results_comparison.keys())
rmse_values = list(results_comparison.values())
train_rmse_values = [
    baseline_train_rmse.mean(),
    train_rmse_scores.mean(),
    dt_train_rmse.mean()
]
test_rmse_values = [
    baseline_rmse.mean(),
    linear_rmse,
    dt_rmse.mean()
]

# Calculate improvements
baseline_rmse_val = baseline_rmse.mean()
improvements = [
    0,  # Baseline has no improvement over itself
    ((baseline_rmse_val - linear_rmse) / baseline_rmse_val) * 100,
    ((baseline_rmse_val - dt_rmse.mean()) / baseline_rmse_val) * 100
]

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# Plot 1: RMSE Comparison Bar Chart
ax1 = plt.subplot(2, 3, 1)
bars = ax1.bar(range(len(model_names)), rmse_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax1.set_xlabel('Model', fontsize=11, fontweight='bold')
ax1.set_ylabel('RMSE', fontsize=11, fontweight='bold')
ax1.set_title('Test RMSE Comparison Across Models', fontsize=12, fontweight='bold')
ax1.set_xticks(range(len(model_names)))
ax1.set_xticklabels([name.replace(' (', '\n(') for name in model_names], rotation=0, ha='center', fontsize=9)
ax1.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, rmse_values)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{val:.2f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: Train vs Test RMSE
ax2 = plt.subplot(2, 3, 2)
x = np.arange(len(model_names))
width = 0.35
bars1 = ax2.bar(x - width/2, train_rmse_values, width, label='Train RMSE', color='#95E1D3', alpha=0.8)
bars2 = ax2.bar(x + width/2, test_rmse_values, width, label='Test RMSE', color='#F38181', alpha=0.8)
ax2.set_xlabel('Model', fontsize=11, fontweight='bold')
ax2.set_ylabel('RMSE', fontsize=11, fontweight='bold')
ax2.set_title('Train vs Test RMSE (Overfitting Analysis)', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels([name.replace(' (', '\n(') for name in model_names], rotation=0, ha='center', fontsize=9)
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom', fontsize=8)

# Plot 3: Improvement Percentage
ax3 = plt.subplot(2, 3, 3)
colors_improve = ['gray' if imp == 0 else '#4ECDC4' if imp > 0 else '#FF6B6B' for imp in improvements]
bars = ax3.bar(range(len(model_names)), improvements, color=colors_improve, alpha=0.8)
ax3.set_xlabel('Model', fontsize=11, fontweight='bold')
ax3.set_ylabel('Improvement (%)', fontsize=11, fontweight='bold')
ax3.set_title('Improvement Over Baseline (%)', fontsize=12, fontweight='bold')
ax3.set_xticks(range(len(model_names)))
ax3.set_xticklabels([name.replace(' (', '\n(') for name in model_names], rotation=0, ha='center', fontsize=9)
ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax3.grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, improvements)):
    height = bar.get_height()
    if val != 0:
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}%',
                ha='center', va='bottom' if val > 0 else 'top', fontsize=10, fontweight='bold')

# Plot 4: Cross-Validation Fold Results
ax4 = plt.subplot(2, 3, 4)
fold_data = {
    'Baseline': baseline_rmse,
    'Linear (Lag/Rolling)': rmse_scores,
    'Decision Tree': dt_rmse
}
fold_df = pd.DataFrame(fold_data)
fold_df_melted = fold_df.melt(var_name='Model', value_name='RMSE')
sns.boxplot(data=fold_df_melted, x='Model', y='RMSE', ax=ax4, palette=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax4.set_xlabel('Model', fontsize=11, fontweight='bold')
ax4.set_ylabel('RMSE', fontsize=11, fontweight='bold')
ax4.set_title('RMSE Distribution Across CV Folds', fontsize=12, fontweight='bold')
ax4.tick_params(axis='x', rotation=15)
ax4.grid(axis='y', alpha=0.3)

# Plot 5: Overfitting Gap (Train - Test)
ax5 = plt.subplot(2, 3, 5)
overfitting_gaps = [train - test for train, test in zip(train_rmse_values, test_rmse_values)]
gap_pct = [(gap / test) * 100 for gap, test in zip(overfitting_gaps, test_rmse_values)]
colors_gap = ['#FF6B6B' if gap < 0 else '#4ECDC4' for gap in overfitting_gaps]
bars = ax5.bar(range(len(model_names)), gap_pct, color=colors_gap, alpha=0.8)
ax5.set_xlabel('Model', fontsize=11, fontweight='bold')
ax5.set_ylabel('Overfitting Gap (%)', fontsize=11, fontweight='bold')
ax5.set_title('Overfitting Analysis (Train-Test Gap %)', fontsize=12, fontweight='bold')
ax5.set_xticks(range(len(model_names)))
ax5.set_xticklabels([name.replace(' (', '\n(') for name in model_names], rotation=0, ha='center', fontsize=9)
ax5.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
ax5.grid(axis='y', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, gap_pct)):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}%',
            ha='center', va='bottom' if val > 0 else 'top', fontsize=10, fontweight='bold')

# Plot 6: Model Ranking Visualization
ax6 = plt.subplot(2, 3, 6)
sorted_models_plot = sorted(results_comparison.items(), key=lambda x: x[1])
sorted_names = [name for name, _ in sorted_models_plot]
sorted_rmse = [rmse for _, rmse in sorted_models_plot]
colors_rank = ['#FFD93D', '#6BCB77', '#4D96FF'][:len(sorted_names)]
bars = ax6.barh(range(len(sorted_names)), sorted_rmse, color=colors_rank, alpha=0.8)
ax6.set_xlabel('RMSE', fontsize=11, fontweight='bold')
ax6.set_ylabel('Model (Ranked)', fontsize=11, fontweight='bold')
ax6.set_title('Model Ranking (Best to Worst)', fontsize=12, fontweight='bold')
ax6.set_yticks(range(len(sorted_names)))
ax6.set_yticklabels([name.replace(' (', '\n(') for name in sorted_names], fontsize=9)
ax6.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, sorted_rmse)):
    width = bar.get_width()
    ax6.text(width, bar.get_y() + bar.get_height()/2.,
            f'  {val:.2f}',
            ha='left', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()

# Save the plot
plt.savefig('model_accuracy_comparison.png', dpi=300, bbox_inches='tight')
print("\nPlot saved as 'model_accuracy_comparison.png'")

# Show the plot
plt.show()

print("\n" + "="*70)
print("VISUALIZATION COMPLETE")
print("="*70)
print("\nGenerated 6 visualization plots:")
print("  1. Test RMSE Comparison - Bar chart showing RMSE for each model")
print("  2. Train vs Test RMSE - Side-by-side comparison showing overfitting")
print("  3. Improvement Over Baseline - Percentage improvement for each model")
print("  4. CV Fold Distribution - Box plot showing RMSE variability across folds")
print("  5. Overfitting Analysis - Train-Test gap percentage for each model")
print("  6. Model Ranking - Horizontal bar chart ranking models best to worst")
print("\nAll plots saved in: model_accuracy_comparison.png")

