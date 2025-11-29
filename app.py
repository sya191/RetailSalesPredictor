import os
from flask import Flask, request, jsonify, send_from_directory

from predict import train_and_evaluate_models, visualize_models_comparison_table
from predict_sales import train_and_save_model, predict_sales

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(
    __name__,
    static_folder=os.path.join(BASE_DIR, "static"),
    static_url_path="/static",
)

# -----------------------------
# Front-end routes
# -----------------------------
@app.route("/")
@app.route("/index.html")
def index():
    """
    Serve the main front-end page.

    This assumes `index.html` is in the same directory as app.py.
    """
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/style.css")
def style():
    """
    Serve the CSS file referenced by index.html.

    This assumes `style.css` is in the same directory as app.py.
    """
    return send_from_directory(BASE_DIR, "style.css")


# -----------------------------
# /train endpoint
# -----------------------------
@app.route("/train", methods=["POST"])
def train_endpoint():
    """
    Trigger training from the front end ("Train Model" button).

    index.html JS expects JSON with:
      - success (bool)
      - message (str)
      - model_name (str)
      - train_rmse (float)
      - test_rmse (float)
      - metrics (dict)
      - log (str)
      - plot_url (str)
    """
    try:
        csv_path = os.path.join(BASE_DIR, "sales_data.csv")

        # 1) Train production model used for predictions
        train_summary = train_and_save_model(csv_path=csv_path)

        # 2) Run evaluation / comparison and create plot in /static
        plot_path = os.path.join(app.static_folder, "model_accuracy_comparison.png")
        eval_result = train_and_evaluate_models(csv_path=csv_path, plot_path=plot_path)

        # 3) Create comparison table visualization
        table_plot_path = os.path.join(app.static_folder, "model_comparison_table.png")
        visualize_models_comparison_table(csv_path=csv_path, plot_path=table_plot_path)

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

        return jsonify(
            {
                "success": True,
                "message": "Training completed successfully.",
                "model_name": eval_result.get("model_name"),
                "train_rmse": eval_result.get("train_rmse"),
                "test_rmse": eval_result.get("test_rmse"),
                "metrics": eval_result.get("metrics"),
                "log": "\n".join(logs).strip(),
                "plot_url": "/static/model_accuracy_comparison.png",
                "table_url": "/static/model_comparison_table.png",
            }
        )

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# -----------------------------
# /predict endpoint
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict_endpoint():
    """
    Make predictions for a given store, product, and date.

    index.html sends JSON:
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
        "store_id": "...",
        "product_id": "...",
        "predictions": [
          { "date": "YYYY-MM-DD", "units_sold": 123.45 },
          ...
        ]
      }

    - For single day:
      {
        "success": true,
        "prediction_type": "day",
        "store_id": "...",
        "product_id": "...",
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

        # Normalize to your dataset's ID format: S###, P#### or accept existing codes
        def normalize_store(value):
            s = str(value)
            if s.upper().startswith("S"):
                return s.upper()
            return f"S{int(float(s)):03d}"

        def normalize_product(value):
            s = str(value)
            if s.upper().startswith("P"):
                return s.upper()
            return f"P{int(float(s)):04d}"

        store_id = normalize_store(raw_store)
        product_id = normalize_product(raw_product)

        prediction = predict_sales(store_id, product_id, date_str, is_week=is_week)

        if is_week:
            # prediction is a dict { "YYYY-MM-DD": value }
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
            # single float
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
    print("BASE_DIR:", BASE_DIR)
    print("Files in BASE_DIR:", os.listdir(BASE_DIR))
    app.run(debug=True)
