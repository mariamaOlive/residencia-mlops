#########################################################################################
# Alunos:																				#
# 	Laianna Lana Virginio da Silva - llvs2@cin.ufpe.br      							#
#   Liviany Reis Rodrigues - lrr@cin.ufpe.br
# 	Lucas Natan Correia Couri - lncc2@cin.ufpe.br   									#
#   Mariama Celi Serafim de Oliveira - mcso@cin.ufpe.br
# 	Priscilla Amarante de Lima - pal4@cin.ufpe.br   									#
#########################################################################################


#########################################################################################
# Bibliotecas																			#
#########################################################################################
import os
import pickle
import numpy as np
import pandas as pd

import mlflow
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing


#########################################################################################
# Constantes																			#
#########################################################################################
NOME_COLUNAS = ('age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country')


#########################################################################################
# Modelo    																			#
#########################################################################################
logged_model = f'./mlruns/1/e7470dabc0d44c0aa0baf4443c7825ef/artifacts/modelo-random-forest'
model = mlflow.pyfunc.load_model(logged_model)


#########################################################################################
#########################################################################################
def tratamento_faltantes(df):
    ## Printa os atributos com dados faltantes (" ?")
    for coluna in NOME_COLUNAS:
        if len(df[df[coluna] == " ?"]) > 0:
            print(coluna)
            print(len(df[df[coluna] == " ?"]))
    
    ## Tratamento dos dados faltantes:
    atr_faltantes = ["workclass", "occupation", "native-country"]
    for atr in atr_faltantes:
        categorias_atr = df.groupby(atr).sum().index.tolist()
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(categorias_atr)
        df[atr] = df[atr].replace(" ?", np.nan)
        df[atr] = df[atr].interpolate(method = 'pad')


def formatar_entrada(colunas_treino, X_test):

    df = pd.concat([colunas_treino, X_test], axis = 0, ignore_index = True)#.head(1)

    df = df.drop(labels = 0, axis = 0)

    colunas_classes = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

    for coluna in colunas_classes:
        for i in range(0,len(df)):
            value = df[coluna].iloc[i]
            df[f'{coluna}_{value}'].iloc[i] = 1 #fazer isso so pra linha i, e nao pra toda a coluna

    df = df.fillna(0)

    return df


def prepare_features(X_test):

    X_test = pd.json_normalize(X_test)

    # Trata os dados faltantes -> "?"
    tratamento_faltantes(X_test)

    # Lê o arquivo com o nome das colunas geradas no dataset de treino
    colunas_treino = pd.read_csv("./Dados/colunas.csv")

    # Formata o teste para as colunas ficarem iguais ao dataset de treino
    formatar_entrada(colunas_treino, X_test)

    ### Tratamento de variaveis continuas
    colunas = ["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week", "education-num"]
    normalize = MinMaxScaler()
    X_test[colunas] = normalize.fit_transform(X_test[colunas])
 
    ### Dropar colunas e separar X e Y
    #colunas_drop = ["class", "education", "workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    colunas_drop = ["education", "workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]


    X_test = X_test.drop(colunas_drop, axis = 1).to_numpy()

    return X_test


#########################################################################################
#########################################################################################
def predict(features):
    preds = model.predict(features)
    return preds[0]


app = Flask('duration-prediction')


@app.route('/predict', methods = ['POST'])
def predict_endpoint():
    adult = request.get_json()
    print(f"\n\n{adult}\n\n")

    features = prepare_features(adult)
    pred = predict(features)

    result = {
        'class': pred
    }

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug = True, host = '0.0.0.0', port = 9696)
