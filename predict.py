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
# Nome das colunas do dataset
NOME_COLUNAS = ('age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country')


#########################################################################################
# Modelo    																			#
#########################################################################################
# Carregando o modelo random forest
logged_model = f'./mlruns/1/ee1069bf3979424690e52d2d5d9b7915/artifacts/modelo-random-forest'
model = mlflow.pyfunc.load_model(logged_model)


#########################################################################################
#########################################################################################

# Função para colocar a entrada no formato correto
def formatar_entrada(colunas_treino, X_test):

    # Junta o dataset de teste com o nome das colunas do one hot encoder
    df = pd.concat([colunas_treino, X_test], axis = 0, ignore_index = True)

    # Colunas das freatures categoricas
    colunas_classes = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

    # Faz o one hot encoding
    for coluna in colunas_classes:
        for i in range(0,len(df)):
            value = df[coluna].iloc[i]
            df[f'{coluna}_{value}'].iloc[i] = 1

    df = df.fillna(0)

    return df

# Função para preparar as features
def prepare_features(dic_test):

    df_test = pd.json_normalize(dic_test)

    # Lê o arquivo com o nome das colunas geradas no dataset de treino
    colunas_treino = pd.read_csv("./Dados/colunas.csv")

    # Formata o teste para as colunas ficarem iguais ao dataset de treino
    X_test = formatar_entrada(colunas_treino, df_test)

    ### Tratamento de variaveis continuas
    colunas = ["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week", "education-num"]
    normalize = MinMaxScaler()
    X_test[colunas] = normalize.fit_transform(X_test[colunas])
 
    ### Dropar colunas que nao serao usadas
    colunas_drop = ["education", "workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    X_test = X_test.drop(colunas_drop, axis = 1).to_numpy()

    return X_test


#########################################################################################
#########################################################################################
# Função para realizar a predição
def predict(features):
    preds = model.predict(features)
    return int(preds[0])

# Flask
app = Flask('duration-prediction')

# Faz a predição e retorna a classe
@app.route('/predict', methods = ['POST'])
def predict_endpoint():

    # Recebe request em json
    adult = request.get_json()
    print(f"\n\n{adult}\n\n")

    # Prepara as features para predição
    features = prepare_features(adult)
    # Realiza a predição
    pred = predict(features)

    # Retorna o resultado em formato json
    result = {
        'class': pred
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug = True, host = '0.0.0.0', port = 9696)
