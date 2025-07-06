from flask import Flask, request, render_template
import pickle
import numpy as np
import os
import datetime

app = Flask(__name__)

model_path = 'Car_mileage_pred_model_SGD.pkl'
scaler_path = 'scaler.pkl'
poly_path = 'poly_transformer.pkl'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
if not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
if not os.path.exists(poly_path):
    raise FileNotFoundError(f"Polynomial transformer file not found: {poly_path}")

model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))
poly = pickle.load(open(poly_path, 'rb'))

@app.route('/', methods=['GET'])
def home():
    year = datetime.datetime.now().year
    return render_template('index.html', prediction_text=None, year=year)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        origin = int(request.form['origin'])
        cylinders = int(request.form['cylinders'])
        horsepower = float(request.form['horsepower'])

        input_data = np.array([[origin, cylinders, horsepower]])
        input_scaled = scaler.transform(input_data)
        input_poly = poly.transform(input_scaled)
        prediction = model.predict(input_poly)[0]

        prediction_text = f'Estimated Mileage: {prediction:.2f} km/l'
        year = datetime.datetime.now().year
        return render_template('index.html', prediction_text=prediction_text, year=year)
    except Exception as e:
        error_text = f'Error: {str(e)}'
        year = datetime.datetime.now().year
        return render_template('index.html', prediction_text=error_text, year=year)

if __name__ == '__main__':
    app.run(debug=True)
