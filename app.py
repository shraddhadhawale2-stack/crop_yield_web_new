from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model/crop_yield_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        input_data = pd.DataFrame([{
            "Crop_Type": data["Crop_Type"],
            "Region": data["Region"],
            "Soil_Type": data["Soil_Type"],
            "Season": data["Season"],
            "Farm_Area_Hectares": float(data["Farm_Area_Hectares"]),
            "Rainfall_mm": float(data["Rainfall_mm"]),
            "Temperature_C": float(data["Temperature_C"]),
            "Humidity_Percent": float(data["Humidity_Percent"]),
            "Soil_pH": float(data["Soil_pH"]),
            "Nitrogen_kg_ha": float(data["Nitrogen_kg_ha"]),
            "Phosphorus_kg_ha": float(data["Phosphorus_kg_ha"]),
            "Potassium_kg_ha": float(data["Potassium_kg_ha"]),
            "Fertilizer_Used_kg": float(data["Fertilizer_Used_kg"]),
            "Pesticide_Used_Liters": float(data["Pesticide_Used_Liters"]),
            "Irrigation_Method": data["Irrigation_Method"],
            "Sunshine_Hours": float(data["Sunshine_Hours"]),
            "Organic_Farming": data["Organic_Farming"]
        }])

        prediction = model.predict(input_data)[0]

        return jsonify({
            "success": True,
            "prediction": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)