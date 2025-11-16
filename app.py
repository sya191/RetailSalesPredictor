from flask import Flask, request, jsonify, send_from_directory
import os

# These must exist in your refactored modules:
# - predict.train_and_evaluate_models
# - predict_sales.train_and_save_model
# - predict_sales.predict_sales
from predict import train_and_evaluate_models
from predict_sales import train_and_save_model, predict_sales

app = Flask(__name__, static_folder="static", static_url_path="/static")


# -----------------------------
# Front-end routes
# -----------------------------
@app.route("/")
def index():
    """Serve index.html for the SPA front end."""
    return send_from_directory(".", "index.html")


@app.route("/style.css")
def style():
    """Serve the CSS file referenced by index.html."""
    return send_from_directory(".", "style.css")


# -----------------------------
# /train endpoint
# -----------------------------
@app.route("/train", methods=["POST"])
def train_endpoint():
    """
    Trigger training from the front end (Train Model button in index.html).

    - Uses predict_sales.train_and_save_model to train the production model and
      save artifacts (sales_prediction_model.pkl, historical_data.pkl).
    - Uses predict.train_and_evaluate_models to compute metrics and save
      model_accuracy_comparison.png under /static.

    index.html expects JSON like:
    {
      "success": true,
      "message": "Training complete",
      "model_name": "...",
      "train_rmse": 123.45,
      "test_rmse": 234.56,
      "metrics": { ... },
      "log": "text...",
      "plot_url": "/static/model_accuracy_comparison.png"
    }
    """
    try:
        csv_path = "sales_data.csv"

        # 1) Train production model used for predictions
        train_summary = train_and_save_model(csv_path=csv_path)  # from predict_sales.py

        # 2) Run evaluation / comparison and create plot
        plot_path = os.path.join(app.static_folder, "model_accuracy_comparison.png")
        eval_result = train_and_evaluate_models(csv_path=csv_path, plot_path=plot_path)

        # Build log text for the Training Log panel
        logs = []
        if train_summary:
            rows = train_summary.get("rows")
            cols = train_summary.get("cols")
            logs.append(
                f"Production model trained on {rows} rows and {cols} columns "
                f"(predict_sales.train_and_save_model)."
            )

        if eval_result.get("log"):
            logs.append("")
            logs.append(eval_result["log"])

        response = {
            "success": True,
            "message": "Training completed successfully.",
            "model_name": eval_result.get("model_name"),
            "train_rmse": eval_result.get("train_rmse"),
            "test_rmse": eval_result.get("test_rmse"),
            "metrics": eval_result.get("metrics"),
            "log": "\n".join(logs).strip(),
            # index.html checks data.plot_url, then falls back to this path
            "plot_url": "/static/model_accuracy_comparison.png",
        }
        return jsonify(response)

    except Exception as e:
        # index.html: if !data.success it shows error
        return jsonify({"success": False, "message": str(e)}), 500


# -----------------------------
# /predict endpoint
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Make predictions for a given store, product, and date.

    index.html sends:
      {
        "store_id": Number,
        "product_id": Number,
        "date": "YYYY-MM-DD",
        "is_week": true/false
      }

    index.html expects:

    - For week:
      {
        "success": true,
        "prediction_type": "week",
        "predictions": [
          { "date": "YYYY-MM-DD", "units_sold": 123.45 },
          ...
        ]
      }

    - For single day:
      {
        "success": true,
        "prediction_type": "day",
        "date": "YYYY-MM-DD",
        "units_sold": 123.45
      }
    """
    try:
        payload = request.get_json(force=True) or {}

        raw_store = payload.get("store_id")
        raw_product = payload.get("product_id")
        date_str = payload.get("date")
        is_week = bool(payload.get("is_week", False))

        if raw_store is None or raw_product is None or date_str is None:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "store_id, product_id, and date are required.",
                    }
                ),
                400,
            )

        # index.html sends numeric IDs; normalize them to S### / P#### for your dataset
        def normalize_store(value):
            s = str(value)
            if s.upper().startswith("S"):
                return s.upper()
            # numeric -> S001, S002, ...
            return f"S{int(float(s)):03d}"

        def normalize_product(value):
            s = str(value)
            if s.upper().startswith("P"):
                return s.upper()
            # numeric -> P0001, P0002, ...
            return f"P{int(float(s)):04d}"

        store_id = normalize_store(raw_store)
        product_id = normalize_product(raw_product)

        # Use high-level helper from predict_sales.py
        prediction = predict_sales(store_id, product_id, date_str, is_week=is_week)

        if is_week:
            # prediction is a dict { "YYYY-MM-DD": value, ... }
            predictions_array = [
                {"date": d, "units_sold": float(v)}
                for d, v in sorted(prediction.items())
            ]
            return jsonify(
                {
                    "success": True,
                    "prediction_type": "week",
                    "store_id": store_id,
                    "product_id": product_id,
                    "predictions": predictions_array,
                }
            )
        else:
            # prediction is a single float
            return jsonify(
                {
                    "success": True,
                    "prediction_type": "day",
                    "store_id": store_id,
                    "product_id": product_id,
                    "date": date_str,
                    "units_sold": float(prediction),
                }
            )

    except FileNotFoundError as e:
        # If model files don't exist yet, tell the front-end to run training first
        return (
            jsonify(
                {
                    "success": False,
                    "message": f"Model artifacts not found: {e}. "
                               f"Train the model first using the 'Train Model' button.",
                }
            ),
            500,
        )
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


if __name__ == "__main__":
    # For local development
    app.run(debug=True)
