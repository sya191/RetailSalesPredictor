import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import warnings
warnings.filterwarnings('ignore')

# Training functions for endpoints and scripts
def train_and_save_model(csv_path="sales_data.csv"):
    # training hte decision tree on the full dataset and saving the results
    # returns a small summary of the data used, the traied model and enginerred
    # features will be saved to disk
    print("Loading and preparing data...")
    data = pd.read_csv(csv_path)
    
    # Drop leakage columns
    leakage_cols = ["Demand", "Units Ordered"]
    data = data.drop(columns=[col for col in leakage_cols if col in data.columns])
    
    # Convert date to datetime
    data["Date"] = pd.to_datetime(data["Date"])
    
    # Sort by date for proper time series features
    data = data.sort_values(["Store ID", "Product ID", "Date"]).reset_index(drop=True)
    
    # Extract date features
    data["DayOfWeek"] = data["Date"].dt.dayofweek
    data["Month"] = data["Date"].dt.month
    data["Year"] = data["Date"].dt.year
    data["DaysSinceStart"] = (data["Date"] - data["Date"].min()).dt.days
    
    
    # creating lag +rolling features
    print("Creating lag and rolling features...")
    
    # Lag features
    data["Lag_1"] = data.groupby(["Store ID", "Product ID"])["Units Sold"].shift(1)
    data["Lag_7"] = data.groupby(["Store ID", "Product ID"])["Units Sold"].shift(7)
    data["Lag_14"] = data.groupby(["Store ID", "Product ID"])["Units Sold"].shift(14)
    data["Lag_30"] = data.groupby(["Store ID", "Product ID"])["Units Sold"].shift(30)
    
    # Rolling statistics
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
    
    # Fill NaN values
    for col in data.columns:
        if 'Rolling' in col or 'Lag' in col:
            data[col] = data.groupby(["Store ID", "Product ID"])[col].ffill()
            data[col] = data[col].fillna(0)
    
    # Drop rows where Lag_1 is missing
    data = data.dropna(subset=["Lag_1"])
    
    # Fill any remaining NaN
    numeric_cols_before = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols_before] = data[numeric_cols_before].fillna(0)
    
    print(f"Data prepared: {data.shape[0]} rows, {data.shape[1]} columns")
    
    #preparing the features and setting the target    
    y = data["Units Sold"]
    X = data.drop(columns=["Units Sold", "Date"])
    
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    
    print(f"Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")
    
    
    #train the model on all data     
    print("\nTraining Decision Tree model on all data...")
    
    # Preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False), categorical_cols),
            ("num", StandardScaler(), numeric_cols)
        ],
        remainder='drop'
    )
    
    # Decision Tree model
    dt_model = DecisionTreeRegressor(
        max_depth=15,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    
    # Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", dt_model)
    ])
    
    # Train on all data
    pipeline.fit(X, y)
    
    print("Model trained successfully!")
    
    # Save the model
    with open("sales_prediction_model.pkl", "wb") as f:
        pickle.dump(pipeline, f)
    
    # Save the data for reference (needed for creating lag features for future predictions)
    data.to_pickle("historical_data.pkl")
    
    print("Model and historical data saved!")
    return {"rows": int(data.shape[0]), "cols": int(data.shape[1])}


# Prediction functions

def _get_last_known_values(store_id, product_id, historical_data):
    # This is a helper function to get the last known values for a store-product combination.
    # Used to provide baseline feature values when generating future predictions.
    # Returns:pd.Series which is a row representing the last known data point for the given store and product.
    store_product_data = historical_data[
        (historical_data["Store ID"] == store_id) &
        (historical_data["Product ID"] == product_id)
    ].sort_values("Date")
    
    if len(store_product_data) == 0:
        raise ValueError(f"No historical data found for Store {store_id}, Product {product_id}")
    
    last_row = store_product_data.iloc[-1]
    return last_row


def _create_features_for_date(store_id, product_id, target_date, historical_data):
    # building a feature wor for a given future da
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    #rolling forward the last known store–product record.
    last_row = _get_last_known_values(store_id, product_id, historical_data)
    
    # Copy the last row and update date
    feature_row = last_row.copy()
    feature_row["Date"] = target_date
    
    # Update date-based features
    feature_row["DayOfWeek"] = target_date.dayofweek
    feature_row["Month"] = target_date.month
    feature_row["Year"] = target_date.year
    feature_row["DaysSinceStart"] = (target_date - historical_data["Date"].min()).days
    
    # Update lag features by shifting from historical data
    store_product_data = historical_data[
        (historical_data["Store ID"] == store_id) &
        (historical_data["Product ID"] == product_id)
    ].sort_values("Date")
    
    # Compute new lag values
    recent_sales = store_product_data["Units Sold"].values
    feature_row["Lag_1"] = recent_sales[-1] if len(recent_sales) >= 1 else 0
    feature_row["Lag_7"] = recent_sales[-7] if len(recent_sales) >= 7 else feature_row["Lag_1"]
    feature_row["Lag_14"] = recent_sales[-14] if len(recent_sales) >= 14 else feature_row["Lag_7"]
    feature_row["Lag_30"] = recent_sales[-30] if len(recent_sales) >= 30 else feature_row["Lag_14"]
    
    # Update rolling features using available historical data
    rolling_values = recent_sales.tolist()
    price_values = store_product_data["Price"].values.tolist()
    inventory_values = store_product_data["Inventory Level"].values.tolist()
    
    # Update rolling statistics
    for window in [7, 14, 30]:
        window_vals = rolling_values[-window:] if len(rolling_values) >= window else rolling_values
        feature_row[f"Rolling_{window}_mean"] = np.mean(window_vals) if len(window_vals) > 0 else 0
        feature_row[f"Rolling_{window}_std"] = np.std(window_vals) if len(window_vals) > 1 else 0
    
    # Update price rolling
    feature_row["Rolling_Price_7"] = np.mean(price_values[-7:])
    feature_row["PriceChange_7"] = feature_row["Price"] - feature_row["Rolling_Price_7"]
    
    # Update inventory rolling
    feature_row["Rolling_Inventory_7"] = np.mean(inventory_values[-7:])
    feature_row["InventoryChange_7"] = feature_row["Inventory Level"] - feature_row["Rolling_Inventory_7"]
    
    # Remove Units Sold and Date (target for prediction, not a feature)
    feature_row = feature_row.drop(["Units Sold", "Date"])
    
    # Convert to DataFrame
    feature_df = feature_row.to_frame().T
    return feature_df


def predict_sales_for_date(store_id, product_id, target_date, model, historical_data):
   # Predict the number of units a store porduct will sell on a given date.
   # Done by building required feature row and passing it through the train model
    feature_df = _create_features_for_date(store_id, product_id, target_date, historical_data)
    prediction = model.predict(feature_df)[0]
    return max(0.0, float(prediction))


def predict_sales_for_week(store_id, product_id, week_start_date, model, historical_data):
    
    # Predict sales for a full week recursively.
    # Where each day's prediction is appended back into the historical_data
    # So that lag features update properly for the next day.
    if isinstance(week_start_date, str):
        week_start_date = pd.to_datetime(week_start_date)

    # creating a copy so we don't overwrite the historical data
    hist = historical_data.copy()
    
    # Filters the store+product to speed things up
    store_product_hist = hist[
        (hist["Store ID"] == store_id) &
        (hist["Product ID"] == product_id)
    ].sort_values("Date").copy()

    if store_product_hist.empty:
        raise ValueError(f"No historical data for {store_id}, {product_id}")

    predictions = {}

    # Extract static features that don't change in the future (price, inventory, etc.)
    last_known_price = float(store_product_hist["Price"].iloc[-1])
    last_known_inventory = float(store_product_hist["Inventory Level"].iloc[-1])

    for i in range(7):
        current_date = week_start_date + pd.Timedelta(days=i)
        date_str = current_date.strftime("%Y-%m-%d")

        # Recursively create feature row for this day 
        feature_row = _create_features_for_date(store_id, product_id, current_date, hist)

        # prediction
        pred = model.predict(feature_row)[0]
        pred = max(0.0, float(pred))
        predictions[date_str] = pred

        # Append prediction to history
        new_row = {
            "Store ID": store_id,
            "Product ID": product_id,
            "Date": current_date,
            "Units Sold": pred,
            "Price": last_known_price,
            "Inventory Level": last_known_inventory,
        }

        hist = pd.concat([hist, pd.DataFrame([new_row])], ignore_index=True)

    return predictions


def load_model_and_data():
    # Loads the train model and historical data from disk
    with open("sales_prediction_model.pkl", "rb") as f:
        model = pickle.load(f)
    historical_data = pd.read_pickle("historical_data.pkl")
    #Returns the saved prediction pipeline and the historical DataFrame,
    return model, historical_data


def predict_sales(store_id, product_id, date_or_week, is_week=False):

    #Main prediction function for front-end integration.
    #Load the trained model and historical data from disk.  
    #Returns the saved prediction pipeline and the historical DataFrame based on user input on weeks/days
    model, historical_data = load_model_and_data()
    
    if is_week:
        return predict_sales_for_week(store_id, product_id, date_or_week, model, historical_data)
    else:
        return predict_sales_for_date(store_id, product_id, date_or_week, model, historical_data)
    

def predict_and_recommend(
    store_id,
    product_id,
    date_str,
    existing_stock,
    is_week=False,
    safety_stock=0,
    incoming_stock=0
):
    # Make a daily or weekly sales prediction and compute the recommended order
    # Quantity, factoring in existing, incoming, and safety stock
    # Returns either a single-day prediction or a 7-day forecast along with the suggested order amount.
    # Load model + historical data
    model, historical_data = load_model_and_data()


    # weekly prediction 
    if is_week:
        week_pred = predict_sales_for_week(store_id, product_id, date_str, model, historical_data)

        total_predicted_demand = sum(week_pred.values())
        inventory_position = existing_stock + incoming_stock

        order_qty = max(total_predicted_demand + safety_stock - inventory_position, 0.0)

        return {
            "success": True,
            "prediction_type": "week",
            "store_id": store_id,
            "product_id": product_id,
            "predictions": week_pred,
            "total_predicted_demand": total_predicted_demand,
            "recommended_order_qty": order_qty,
        }

    # single day prediction
    else:
        pred = predict_sales_for_date(store_id, product_id, date_str, model, historical_data)

        inventory_position = existing_stock + incoming_stock
        order_qty = max(pred + safety_stock - inventory_position, 0.0)

        return {
            "success": True,
            "prediction_type": "day",
            "store_id": store_id,
            "product_id": product_id,
            "date": date_str,
            "units_predicted": pred,
            "recommended_order_qty": order_qty,
        }