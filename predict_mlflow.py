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
logged_model = f'./mlruns/1/e7470dabc0d44c0aa0baf4443c7825ef/artifacts/modelo-random-forest'
model = mlflow.pyfunc.load_model(logged_model)


#########################################################################################
#########################################################################################
# Função para tratar dados faltantes 
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

    return df

# Função para colocar a entrada no formato correto
def formatar_entrada(colunas_treino, X_test):

    # Junta o dataset de teste com o nome das colunas do one hot encoder
    df = pd.concat([colunas_treino, X_test], axis = 0, ignore_index = True)

    # Colunas das categoricas
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

    # Trata os dados faltantes -> "?"
    df_test = tratamento_faltantes(df_test)

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
    adult = request.get_json()
    print(f"\n\n{adult}\n\n")

    features = prepare_features(adult)
    pred = predict(features)

    result = {
        'class': pred
    }

    print("\n\n{pred}\n\n")

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug = True, host = '0.0.0.0', port = 9696)
