import os
import pickle
import numpy as np
from flask import Flask, render_template, request

app = Flask(__name__)

# File paths
MODEL_PATH = 'Car_mileage_pred_model_SGD.pkl'
SCALER_PATH = 'scaler.pkl'

# Initialize model and scaler
model = None
scaler = None

# Load model and scaler
try:
    with open(MODEL_PATH, 'rb') as model_file:
        model = pickle.load(model_file)
    with open(SCALER_PATH, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print("✅ Model and scaler loaded successfully.")
except FileNotFoundError:
    print(f"❌ Error: '{MODEL_PATH}' or '{SCALER_PATH}' not found.")
except Exception as e:
    print(f"❌ Error loading model/scaler: {e}")

# Utility function for safe float conversion
def safe_float(value, field_name):
    try:
        return float(value.strip())
    except Exception:
        raise ValueError(f"Invalid input for '{field_name}'.")

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        if model is None or scaler is None:
            prediction = "Error: Model or scaler not loaded."
        else:
            try:
                form = request.form
                print("📥 Received form data:", form)

                # Convert form inputs to float
                cylinders = safe_float(form.get('cylinders', ''), 'cylinders')
                displacement = safe_float(form.get('displacement', ''), 'displacement')
                horsepower = safe_float(form.get('horsepower', ''), 'horsepower')
                weight = safe_float(form.get('weight', ''), 'weight')
                acceleration = safe_float(form.get('acceleration', ''), 'acceleration')
                model_year = safe_float(form.get('model_year', ''), 'model_year')
                origin = safe_float(form.get('origin', ''), 'origin')

                # Create input array
                input_data = np.array([[cylinders, displacement, horsepower,
                                        weight, acceleration, model_year, origin]])
                print("📊 Input data before scaling:", input_data)

                # Scale and predict
                scaled_input = scaler.transform(input_data)
                print("📐 Scaled input:", scaled_input)

                mpg = model.predict(scaled_input)[0]
                prediction = f"{round(mpg, 2)}"
                print("🚗 Predicted MPG:", prediction)

            except ValueError as ve:
                prediction = f"Error: {ve}"
                print("⚠️", ve)
            except Exception as e:
                prediction = f"Unexpected error during prediction: {e}"
                print("❌", e)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    print(f"📁 Running from directory: {os.getcwd()}")
    if model is not None and scaler is not None:
        app.run(debug=True)
    else:
        print("🚫 App not started. Model or scaler failed to load.")
