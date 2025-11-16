# app.py
from flask import Flask, render_template, request
from predict import train_inventory_model as train_model

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    metrics = None
    top_features = None

    if request.method == "POST":
        # Train the model when the button is pressed
        metrics_raw, top_features_df = train_model("retail_store_inventory.csv")

        # Convert numpy / pandas objects into JSON-friendly Python types
        metrics = {
            "rmse_scores": [float(v) for v in metrics_raw["rmse_scores"]],
            "mean_rmse": float(metrics_raw["mean_rmse"]),
        }
        top_features = top_features_df.to_dict(orient="records")

    return render_template("index.html", metrics=metrics, top_features=top_features)


if __name__ == "__main__":
    # Run the development server
    app.run(debug=True)
