import os
import pickle
import numpy as np
import pandas as pd

import mlflow
from flask import Flask, request, jsonify


logged_model = f'./mlruns/1/e7470dabc0d44c0aa0baf4443c7825ef/artifacts/modelo-random-forest'
model = mlflow.pyfunc.load_model(logged_model)

def onehot_encoder(df):

    colunas_cat = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    
    for coluna in colunas_cat:

        #print(coluna)
        df_coluna = pd.get_dummies(df[coluna], prefix=coluna)
        df = df.join(df_coluna)
    
    return df


def prepare_features(X_test):

    ### Tratamento de variaveis categoricas
    X_test = onehot_encoder(X_test)

    ### Tratamento de variaveis continuas
    normalize = MinMaxScaler()
    
    colunas = ["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week", "education-num"]

    X_test[colunas] = normalize.fit_transform(X_test[colunas])
 
    ### Dropar colunas e separar X e Y
    colunas_drop = ["class", "education", "workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

    X_train = df_train.drop(colunas_drop, axis = 1).to_numpy()
    y_train = df_train["class"].values
    #X_val = df_valid.drop(colunas_drop, axis = 1).to_numpy()
    #y_val = df_valid["class"].values

    label_encoder = preprocessing.LabelEncoder()

    y_train = label_encoder.fit_transform(y_train)


    '''features = {}
    
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'], ride['DOLocationID']) #f"{ride['PULocationID']}_{ride['DOLocationID']}"
    
    features['trip_distance'] = ride['trip_distance']
    features = pd.json_normalize(features)'''
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
