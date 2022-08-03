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
SEED = 42


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
def onehot_encoder(df):

    colunas_cat = ["workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    
    for coluna in colunas_cat:

        #print(coluna)
        df_coluna = pd.get_dummies(df[coluna], prefix=coluna)
        df = df.join(df_coluna)
    
    return df


@task
def add_features(df_train): #, df_val

    print(f"\n\nTamanho do treino: {len(df_train)}\n\n")
    #print(f"Tamanho do valid: {len(df_val)}")

    ### Tratamento de variaveis categoricas
    df_train = onehot_encoder(df_train)
    #df_val = onehot_encoder(df_val)

    coluna = 'native-country_ Holand-Netherlands'
    df_train[coluna] = 0
    #df_val[coluna] = 0


    ### Tratamento de variaveis continuas
    colunas = ["age", "fnlwgt", "capital-gain", "capital-loss", "hours-per-week", "education-num"]
    normalize = MinMaxScaler()
    df_train[colunas] = normalize.fit_transform(df_train[colunas])
    #df_val[colunas] = normalize.fit_transform(df_val[colunas])

    ### Dropar colunas e separar X e Y
    colunas_drop = ["class", "education", "workclass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

    X_train = df_train.drop(colunas_drop, axis = 1)
    
    #X_train[:1].to_csv("./Dados/colunas.csv", index = False)
        
    X_train = X_train.to_numpy()
    y_train = df_train["class"].values
    #X_val = df_valid.drop(colunas_drop, axis = 1).to_numpy()
    #y_val = df_valid["class"].values

    label_encoder = preprocessing.LabelEncoder()

    y_train = label_encoder.fit_transform(y_train)
    #y_val = label_encoder.fit_transform(y_val)

    return X_train, y_train
    #return X_train, X_val, y_train, y_val


#########################################################################################
#########################################################################################
@task
def train_model_search(X_train, y_train):

    def objective(params):
    
        with mlflow.start_run():
            mlflow.set_tag("modelo", "random_forest")
            mlflow.log_params(params)

            clf = RandomForestClassifier(**params)

            accuracy = cross_val_score(clf, X_train, y_train, cv = 5).mean()

            mlflow.log_metric("acuracia", accuracy)
            
            # Log the model created by this run.
            mlflow.sklearn.log_model(clf, "modelo-random-forest") 
        
        return {'loss': -accuracy, 'status': STATUS_OK}                                                     

    search_space = {
                    'max_depth': hp.randint('max_depth', 10, 200),
                    'n_estimators': hp.randint('n_estimators', 200, 1000),
                    'criterion': hp.choice('criterion', ['gini','entropy']),
                    'random_state': SEED
                   }

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
@task
def train_best_model(X_train, y_train, X_test, y_test, best_params):

    with mlflow.start_run():
        
        mlflow.log_params(best_params)

        clf = RandomForestClassifier(**best_params)

        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)

        mlflow.log_metric("acuracia", accuracy)

        mlflow.sklearn.log_model(clf, "modelo-random-forest")       



#########################################################################################
#########################################################################################
@task
def model_regitry():
    
    client = MlflowClient(tracking_uri = MLFLOW_TRACKING_URI)
    
    runs = client.search_runs(
                              experiment_ids = '1',
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
    #print(f"\n\nVersão: {model_version}\n\n")

    new_stage = "Production"
    client.transition_model_version_stage(
                                          name = model_name,
                                          version = model_version,
                                          stage = new_stage,
                                          archive_existing_versions = True
                                         )

    date = datetime.today().date()    
    client.update_model_version(
                                name = model_name,
                                version = model_version,
                                description = f"O modelo na versão {model_version} mudou para {new_stage} em {date}"
                               )


#########################################################################################
#########################################################################################
@flow(task_runner=SequentialTaskRunner())
def main(train_path ="Dados/adult.data", test_path ="Dados/adult.test"):
    
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("census-experiment")
    
    df_train = read_dataframe(train_path)
    df_test = read_dataframe(test_path, 1)
   
    X_train, y_train = add_features(df_train)
    X_test, y_test = add_features(df_test)
 
    best_params = train_model_search(X_train, y_train)
    train_best_model(X_train, y_train, X_test, y_test, best_params)

    model_regitry()


print("antes")
main()
print("depois")