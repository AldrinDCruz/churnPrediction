from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np

# Load the trained model from disk
model = joblib.load('bank_churn_model.pkl')  # Adjust the path as needed

app = Flask(__name__)

# Route to serve the HTML page
@app.route('/')
def home():
    return render_template('index.html')  # This will serve the HTML file

# API endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the POST request
        data = request.get_json()  # Expecting JSON with 'features' as key
        
        # Convert the data to numpy array (model expects numerical input)
        features = np.array(data['features']).reshape(1, -1)  # Reshape to make it 2D array
        
        # Make prediction
        prediction = model.predict(features)

        # Return the result as a JSON response
        return jsonify({'churn_prediction': int(prediction[0])})

    except Exception as e:
        # Return an error message if something goes wrong
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
