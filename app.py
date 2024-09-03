from flask import Flask, request, jsonify
from joblib import load
import pandas as pd
from nlp_model import parse

app = Flask(__name__)
model = load('shipping_cost_predictor.pkl')  # Load your pre-trained model
preprocessor = load('preprocessor.pkl')  # Load the pre-trained preprocessor used during the training

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.json['query']
    parsed_data = parse(user_input)

    # Create a DataFrame from the parsed data
    df = pd.DataFrame([parsed_data])

    # Transform the DataFrame using the loaded preprocessor
    df_transformed = pd.DataFrame(preprocessor.transform(df), columns=preprocessor.get_feature_names_out())

    # Use the model to predict the shipment cost
    prediction = model.predict(df_transformed)[0]  # Assuming model.predict returns an array

    # Get the carrier from the transformed DataFrame, and make sure it is serializable
    carrier_series = df_transformed.filter(regex='^cat__Carrier_').idxmax(axis=1)
    carrier = carrier_series.iloc[0].split('__')[-1] if not carrier_series.empty else "Unknown Carrier"

    # Prepare the response, ensuring all data is JSON serializable
    response = {
        'Carrier': carrier,
        'Carbon Emission Rate g per mile': float(parsed_data['Carbon Emission Rate g per mile']),  # Ensure float type
        'cost': float(prediction),  # Convert NumPy types to Python native type
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
