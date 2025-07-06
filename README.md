#  Vehicle Mileage Predictor (SGD Model)

This project is a machine learning web application that predicts the mileage (kilometers per liter) of a vehicle using **Stochastic Gradient Descent Regression (SGDRegressor)**. It's designed for users who want to estimate fuel efficiency based on engine and vehicle features.

![Mileage Prediction](https://img.shields.io/badge/Model-SGDRegressor-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![Flask](https://img.shields.io/badge/Flask-WebApp-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

---

## 📌 Features

* Predicts mileage (`km/l`) using:

  * Origin
  * Cylinders
  * Displacement
  * Horsepower
  * Weight
  * Acceleration
  * Model Year
* Trained using `SGDRegressor` from **scikit-learn**
* VIF and EDA analysis included in Jupyter Notebook
* Web app interface with responsive UI (Tailwind CSS)
* Fast model prediction through Flask backend

---

## 🧠 Tech Stack

| Component         | Technology                       |
| ----------------- | -------------------------------- |
| **Frontend**      | HTML, Tailwind CSS, Font Awesome |
| **Backend**       | Flask, Python                    |
| **ML Model**      | Scikit-learn (SGDRegressor)      |
| **Visualization** | matplotlib, seaborn              |
| **Deployment**    | Localhost (Flask)                |

---

## 📂 File Structure

```
├── app.py                      # Flask web application
├── templates/
│   └── index.html              # Frontend interface
├── static/
│   └── style.css (optional)    # Additional styling
├── Car_milage_prediction_SGD.ipynb  # Model training + EDA
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Omiiii1221/vehicle-mileage-predictor.git
cd vehicle-mileage-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Flask App

```bash
python app.py
```

### 4. Open in Browser

Go to: `http://127.0.0.1:5000`

---

## 🧪 Sample Inputs

| Feature      | Example Value |
| ------------ | ------------- |
| Origin       | 1             |
| Cylinders    | 4             |
| Displacement | 140           |
| Horsepower   | 90            |
| Weight       | 2264          |
| Acceleration | 15.5          |
| Model Year   | 82            |

---

## 📈 Model Insights

* **Algorithm**: SGDRegressor
* **Evaluation**: R² score, RMSE, and VIF calculated
* **Preprocessing**: StandardScaler applied
* **Exploration**: Correlation matrix, VIF, and EDA included

---

## 📌 To-Do / Future Enhancements

* ✅ Add model performance metrics to the UI
* ⏳ Deploy on Render / Heroku
* ⏳ Add more models (Ridge, Lasso, Random Forest)
* ⏳ Allow users to upload a CSV for bulk predictions

---

## 🙋‍♂️ Author

**Om Gaikwad**
📧 [gaikwadom465@gmail.com](mailto:gaikwadom465@gmail.com)
🔗 [LinkedIn](https://www.linkedin.com/in/om-gaikwad-a70421310/)
🐙 [GitHub](https://github.com/Omiiii1221)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
