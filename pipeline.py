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
import numpy as np
import pandas as pd

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import space_eval

import mlflow

from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from datetime import datetime


#########################################################################################
# Constantes																			#
#########################################################################################
NOME_COLUNAS = ('age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'class')
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "census-experiment"
SEED = 42


#########################################################################################
#########################################################################################
# Função para tratar os dados faltantes
def tratamento_faltantes(df):
    ## Printa os atributos com dados faltantes (" ?")
    '''for coluna in NOME_COLUNAS:
        if len(df[df[coluna] == " ?"]) > 0:
            print(coluna)
            print(len(df[df[coluna] == " ?"]))'''
    
    ## Tratamento dos dados faltantes:
    atr_faltantes = ["workclass", "occupation", "native-country"]
    for atr in atr_faltantes:
        categorias_atr = df.groupby(atr).sum().index.tolist()
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(categorias_atr)
        df[atr] = df[atr].replace(" ?", np.nan)
        df[atr] = df[atr].interpolate(method = 'pad')

# Leitura do dataframe
@task
def read_dataframe(filename, skip_num = 0):

    df = pd.read_csv(filename, names = NOME_COLUNAS, index_col = False, skiprows = skip_num)

    df['workclass'] = df['workclass'].astype('category')
    df['education'] = df['education'].astype('category')
    df['marital-status'] = df['marital-status'].astype('category')
    df['occupation'] = df['occupation'].astype('category')
    df['relationship'] = df['relationship'].astype('category')
    df['race'] = df['race'].astype('category')
    df['sex'] = df['sex'].astype('category')
    df['native-country'] = df['native-country'].astype('category')
    df['class'] = df['class'].astype('category')

    df.drop_duplicates(inplace = True)

    tratamento_faltantes(df)

    return df


#########################################################################################
#########################################################################################
#Função para codificar as colunas categoricas (one hot encoding)
def onehot_encoder(df):

    colunas_cat = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    
    for coluna in colunas_cat:

        #print(coluna)
        df_coluna = pd.get_dummies(df[coluna], prefix=coluna)
        df = df.join(df_coluna)
    
    return df

# Tratamento das features categoricas e continuas
@task
def add_features(df_train): 

    print(f"\n\nTamanho do treino: {len(df_train)}\n\n")

    ### Tratamento de variaveis categoricas
    df_train = onehot_encoder(df_train)

    coluna = 'native-country_ Holand-Netherlands'
    df_train[coluna] = 0

    ### Tratamento de variaveis continuas
    colunas = ["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week", "education-num"]
    normalize = MinMaxScaler()
    df_train[colunas] = normalize.fit_transform(df_train[colunas])

    ### Dropar colunas e separar X e Y
    colunas_drop = ["class", "education", "workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

    X_train = df_train.drop(colunas_drop, axis = 1)
    X_train = X_train.to_numpy()
    y_train = df_train["class"].values

    label_encoder = preprocessing.LabelEncoder()

    y_train = label_encoder.fit_transform(y_train)

    return X_train, y_train


#########################################################################################
#########################################################################################
# Função que faz a busca dos melhores parâmetros
@task
def train_model_search(X_train, y_train):
    # Função objetivo
    def objective(params):
    
        with mlflow.start_run():
            mlflow.set_tag("modelo", "random_forest")
            mlflow.log_params(params)

            clf = RandomForestClassifier(**params)

            accuracy = cross_val_score(clf, X_train, y_train, cv = 5).mean()

            mlflow.log_metric("acuracia", accuracy)
            mlflow.sklearn.log_model(clf, "modelo-random-forest") 
        
        return {'loss': -accuracy, 'status': STATUS_OK}                                                     

    # Espaço de busca
    search_space = {
                    'max_depth': hp.randint('max_depth', 10, 200),
                    'n_estimators': hp.randint('n_estimators', 200, 1000),
                    'criterion': hp.choice('criterion', ['gini','entropy']),
                    'random_state': SEED
                   }

    # Otimização da função objetivo
    best_result = fmin(
                       fn = objective,
                       space = search_space,
                       algo = tpe.suggest,
                       max_evals = 20,
                       trials = Trials()
                      )

    dic_best = space_eval(search_space, best_result)

    return dic_best


#########################################################################################
#########################################################################################
# Treina o modelo com os melhores parâmetros encontrados
@task
def train_best_model(X_train, y_train, X_test, y_test, best_params):

    with mlflow.start_run():
        
        mlflow.log_params(best_params)

        clf = RandomForestClassifier(**best_params)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_metric("acuracia", accuracy)
        mlflow.set_tag("melhor_modelo", "melhor")
        mlflow.set_tag("modelo", "random_forest")
        mlflow.sklearn.log_model(clf, "modelo-random-forest")       


#########################################################################################
#########################################################################################
# Registro de modelos
@task
def model_regitry(experiment_id):
    
    client = MlflowClient(tracking_uri = MLFLOW_TRACKING_URI)
    
    # Procura por runs com o melhor modelo encontrado, de acordo com a tag
    runs = client.search_runs(
                              experiment_ids = experiment_id,
                              filter_string = "tags.melhor_modelo = 'melhor'",
                              run_view_type = ViewType.ACTIVE_ONLY,
                              max_results = 5
                             )

    for run in runs:
        print(f"\n\nrun id: {run.info.run_id}, acuracia: {run.data.metrics['acuracia']:.4f}\n\n")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    model_name = "census-classifier"

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri = model_uri, name = model_name)

    latest_versions = client.get_latest_versions(name = model_name)

    versao = 0
    for version in latest_versions:
        print(f"\n\nversion: {version.version}, stage: {version.current_stage}\n\n")
        if versao < version.version:
            versao = version.version

    model_version = versao

    # Colocando o melhor para produção
    new_stage = "Production"
    client.transition_model_version_stage(
                                          name = model_name,
                                          version = model_version,
                                          stage = new_stage,
                                          archive_existing_versions = True
                                         )

    # Atualizando a descrição do modelo
    date = datetime.today().date()    
    client.update_model_version(
                                name = model_name,
                                version = model_version,
                                description = f"O modelo na versão {model_version} mudou para {new_stage} em {date}"
                               )


#########################################################################################
#########################################################################################
# Pipeline
@flow(task_runner=SequentialTaskRunner())
def main(train_path = "./Dados/adult.data", test_path = "./Dados/adult.test"):
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    current_experiment = dict(mlflow.get_experiment_by_name(EXPERIMENT_NAME))
    experiment_id = current_experiment['experiment_id']
    
    df_train = read_dataframe(train_path)
    df_test = read_dataframe(test_path, 1)
   
    X_train, y_train = add_features(df_train)
    X_test, y_test = add_features(df_test)
 
    best_params = train_model_search(X_train, y_train)
    train_best_model(X_train, y_train, X_test, y_test, best_params)

    model_regitry(experiment_id)


print("antes")
main()
print("depois")