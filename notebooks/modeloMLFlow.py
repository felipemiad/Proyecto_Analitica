import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error
import pandas as pd

# Cargar los datos
df_definitivo = pd.read_csv("df_definitivo.csv")
variables_finales = ["FAMI_ESTRATOVIVIENDA", "FAMI_PERSONASHOGAR", "FAMI_TIENEINTERNET", 
                     "ESTU_HORASSEMANATRABAJA", "FAMI_COMECARNEPESCADOHUEVO", "COLE_NATURALEZA", "ESTU_DEPTO_RESIDE","COLE_JORNADA","COLE_GENERO"]

# Configurar la URI de MLflow (IP de tu instancia EC2)
mlflow.set_tracking_uri("http://54.161.153.33:5000")

# Definir el nombre del experimento
mlflow.set_experiment("ICFES_Experiment")

# Definir funciones para procesamiento y modelado
def dividir_datos(dataframe):
    X = dataframe[variables_finales]
    y = dataframe["PUNT_GLOBAL"]
    return X, y

def crear_train_test(tupla):
    X, y = tupla
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    oe = OrdinalEncoder()
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc, y_train, y_test

datos = dividir_datos(df_definitivo)
datos2 = crear_train_test(datos)
X_test, y_test = datos2[1], datos2[3]

def calcular_mse(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse

# Iniciar el experimento principal
with mlflow.start_run() as run:
    
    # Modelo Random Forest
    with mlflow.start_run(nested=True):
        modelo_rf = RandomForestRegressor(n_estimators=200, max_depth=4, max_features=5, random_state=0)
        modelo_rf.fit(datos2[0], datos2[2])
        mse_rf = calcular_mse(modelo_rf, X_test, y_test)
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 4)
        mlflow.log_param("max_features", 5)
        mlflow.log_metric("mse", mse_rf)
        mlflow.sklearn.log_model(modelo_rf, "model_random_forest")
        print(f"Random Forest - MSE: {mse_rf}")

    # Modelo Linear Regression
    with mlflow.start_run(nested=True):
        modelo_lineal = LinearRegression()
        modelo_lineal.fit(datos2[0], datos2[2])
        mse_lineal = calcular_mse(modelo_lineal, X_test, y_test)
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_metric("mse", mse_lineal)
        mlflow.sklearn.log_model(modelo_lineal, "model_linear_regression")
        print(f"Linear Regression - MSE: {mse_lineal}")

    # Modelo Gradient Boosting Regressor
    with mlflow.start_run(nested=True):
        modelo_gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=0)
        modelo_gbr.fit(datos2[0], datos2[2])
        mse_gbr = calcular_mse(modelo_gbr, X_test, y_test)
        mlflow.log_param("model_type", "GradientBoostingRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_depth", 3)
        mlflow.log_metric("mse", mse_gbr)
        mlflow.sklearn.log_model(modelo_gbr, "model_gradient_boosting")
        print(f"Gradient Boosting - MSE: {mse_gbr}")


    

