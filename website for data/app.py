import pickle
import numpy as np
from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
                                                
# ========== LOAD MODEL AND SCALER ==========
model = pickle.load(open("fraud_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ========== HOME PAGE ==========
@app.route('/')
def home():
    return render_template('index.html')

# ========== PREDICTION ROUTE ==========
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read input features
        features = [float(x) for x in request.form.values()]
        final = np.array(features).reshape(1, -1)

        # Scale input
        final_scaled = scaler.transform(final)

        # Predict
        prediction = model.predict(final_scaled)

        result = "⚠️ Fraudulent Transaction Detected!" if prediction[0] == 1 else "✅ Legitimate Transaction"

        return render_template('index.html', prediction_text=result)

    except:
        return render_template('index.html', prediction_text="Invalid input!")

# Run server
if __name__ == "__main__":
    app.run(debug=True)



