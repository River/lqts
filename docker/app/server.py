from flask import Flask, jsonify, request
import numpy as np
import tensorflow as tf
from keras.models import load_model

# load the model
model = load_model('lqtsnet')

# create the Flask app
app = Flask(__name__)

# define the endpoint that accepts requests
@app.route('/', methods=['POST'])
def predict():
    # get the input data as a numpy array
    json_data = request.get_json()
    model_input = np.array(json_data)

    # make the prediction using the model
    prediction = model.predict(model_input)

    # return the prediction as a JSON response
    return jsonify({'prediction': prediction.tolist()})

if __name__ == "__main__":
    # run the Flask server
    app.run(host='0.0.0.0', port=8501)