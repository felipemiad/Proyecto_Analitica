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

# Función para dividir los datos en X (características) y y (variable objetivo)
def dividir_datos(dataframe):
    X = dataframe[variables_finales]
    y = dataframe["PUNT_GLOBAL"]
    return X, y

datos = dividir_datos(df_definitivo)

# Función para crear las particiones de train y test, y codificar las variables categóricas
def crear_train_test(tupla):
    X, y = tupla
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    oe = OrdinalEncoder()
    oe.fit(X_train)
    X_train_enc = oe.transform(X_train)
    X_test_enc = oe.transform(X_test)
    return X_train_enc, X_test_enc, y_train, y_test

datos2 = crear_train_test(datos)

# Función para entrenar el modelo de RandomForestRegressor
def crear_modelo_random_forest(tupla, num_trees=100, max_depth=None, max_features='auto'):
    X_train, X_test, y_train, y_test = tupla
    rf_regressor = RandomForestRegressor(n_estimators=num_trees, max_depth=max_depth, max_features=max_features, random_state=0)
    modelo = rf_regressor.fit(X_train, y_train)
    return modelo

# Función para entrenar el modelo de LinearRegression
def crear_modelo_lineal(tupla):
    X_train, X_test, y_train, y_test = tupla
    linear_regressor = LinearRegression()
    modelo = linear_regressor.fit(X_train, y_train)
    return modelo

# Función para entrenar el modelo de GradientBoostingRegressor
def crear_modelo_gradient_boosting(tupla, n_estimators=100, learning_rate=0.1, max_depth=3):
    X_train, X_test, y_train, y_test = tupla
    gbr_regressor = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=0)
    modelo = gbr_regressor.fit(X_train, y_train)
    return modelo

# Función para hacer predicciones
def predecir(modelo, tupla):
    X_train, X_test, y_train, y_test = tupla
    y_pred = modelo.predict(X_test)
    return y_pred

# Función para calcular el MSE (Error Cuadrático Medio)
def calcular_mse(modelo, X_test, y_test):
    y_pred = modelo.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return np.float64(mse)

# Preparar los datos de prueba
datos_internos = crear_train_test(datos)
X_test = datos_internos[1]
y_test = datos_internos[3]

# Establecer el nombre del experimento en MLflow
mlflow.set_experiment("ICFES_Experiment")  # Nombre del experimento

# Iniciar un experimento en MLflow
with mlflow.start_run():
    # Parámetros del modelo RandomForestRegressor
    num_trees = 200  # Número de árboles en el Random Forest
    max_depth = 4  # Profundidad máxima de cada árbol
    max_features = 5  # Características a considerar para cada árbol
    
    # Entrenar el modelo de RandomForestRegressor
    modelo_rf = crear_modelo_random_forest(datos2, num_trees=num_trees, max_depth=max_depth, max_features=max_features)
    
    # Calcular el MSE para RandomForest
    mse_rf = calcular_mse(modelo_rf, X_test, y_test)
    
    # Log de los parámetros y métricas del modelo RandomForest
    mlflow.log_param("model_type", "RandomForestRegressor")
    mlflow.log_param("n_estimators", num_trees)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_features", max_features)
    mlflow.log_metric("mse", mse_rf)
    
    # Log del modelo RandomForest
    mlflow.sklearn.log_model(modelo_rf, "model_random_forest")
    
    # Mostrar los resultados del modelo RandomForest
    print("Random Forest - MSE es de: " + str(mse_rf))
    print("Parámetros del modelo RandomForest:")
    print("Número de Árboles: " + str(num_trees))
    print("Máxima Profundidad: " + str(max_depth))
    print("Máximas Características por Árbol: " + str(max_features))

# Iniciar otro experimento en el mismo run para el modelo LinearRegression
with mlflow.start_run():
    # Entrenar el modelo de Linear Regression
    modelo_lineal = crear_modelo_lineal(datos2)
    
    # Calcular el MSE para el modelo de regresión lineal
    mse_lineal = calcular_mse(modelo_lineal, X_test, y_test)
    
    # Log de los parámetros y métricas del modelo LinearRegression
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse_lineal)
    
    # Log del modelo LinearRegression
    mlflow.sklearn.log_model(modelo_lineal, "model_linear_regression")
    
    # Mostrar los resultados del modelo Linear Regression
    print("Linear Regression - MSE es de: " + str(mse_lineal))

# Iniciar otro experimento para el modelo Gradient Boosting Regressor
with mlflow.start_run():
    # Parámetros del modelo GradientBoostingRegressor
    n_estimators = 100  # Número de estimadores
    learning_rate = 0.1  # Tasa de aprendizaje
    max_depth = 3  # Profundidad máxima de los árboles
    
    # Entrenar el modelo de Gradient Boosting Regressor
    modelo_gbr = crear_modelo_gradient_boosting(datos2, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth)
    
    # Calcular el MSE para Gradient Boosting
    mse_gbr = calcular_mse(modelo_gbr, X_test, y_test)
    
    # Log de los parámetros y métricas del modelo GradientBoosting
    mlflow.log_param("model_type", "GradientBoostingRegressor")
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_metric("mse", mse_gbr)
    
    # Log del modelo GradientBoosting
    mlflow.sklearn.log_model(modelo_gbr, "model_gradient_boosting")
    
    # Mostrar los resultados del modelo Gradient Boosting
    print("Gradient Boosting - MSE es de: " + str(mse_gbr))
    print("Parámetros del modelo Gradient Boosting:")
    print("Número de Estimadores: " + str(n_estimators))
    print("Tasa de Aprendizaje: " + str(learning_rate))
    print("Máxima Profundidad: " + str(max_depth))


    

