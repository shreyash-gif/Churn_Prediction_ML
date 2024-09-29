from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
app = Flask(__name__)


with open('models/best_churn_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('models/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        
        data = request.get_json(force=True)
        input_data = pd.DataFrame([data])
        numeric_features = input_data[['tenure', 'MonthlyCharges', 'TotalCharges']]
        scaled_numeric_features = scaler.transform(numeric_features)
        gender_encoded = np.where(input_data['gender'] == 'Male', 1, 0)

        internet_service_encoded = pd.get_dummies(input_data['InternetService'], drop_first=True)
        contract_encoded = pd.get_dummies(input_data['Contract'], drop_first=True)
        payment_method_encoded = pd.get_dummies(input_data['PaymentMethod'], drop_first=True)

        final_input = np.hstack([
            scaled_numeric_features,
            gender_encoded.reshape(-1, 1),
            internet_service_encoded.values,
            contract_encoded.values,
            payment_method_encoded.values
        ])

        prediction = model.predict(final_input)
        return jsonify({
            'prediction': int(prediction[0]),
            'message': 'Churn prediction: 1 indicates churn, 0 indicates no churn'
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'message': 'Error processing the request.'})

if __name__ == '__main__':
    app.run(debug=True)
