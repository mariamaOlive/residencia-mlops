import os
import pickle
import numpy as np
import pandas as pd

import mlflow
from flask import Flask, request, jsonify


logged_model = f'C:/Users/victo/PycharmProjects/mlops/mlruns/1/4526dbe8bab74998a23b76acf0e9d296/artifacts/models_mlflow'
model = mlflow.pyfunc.load_model(logged_model)


def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    features = pd.json_normalize(features)
    return features


def predict(features):
    preds = model.predict(features)
    return float(preds[0])


app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    ride = request.get_json()

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
