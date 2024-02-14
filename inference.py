# Importar las librerías necesarias
import pandas as pd
import numpy as np
import joblib
# Importar las funciones necesarias desde outils.py
from src.outils import get_features, load_model, load_dataframe, save_dataframe

# Cargar el modelo entrenado
best_model = load_model('artifacts/best_model.pkl')

# Cargar el estado del imputer y del selector
imputer = load_model('data/prep/imputer.pkl')
selector = load_model('data/prep/selector.pkl')

# Definir las características
features = get_features()

# Cargar los datos de predicción
predict_path = 'data/inference/test.csv'
df_predict = load_dataframe(predict_path)

# Preparar los datos de predicción
X_predict = df_predict[features]
X_predict_imputed = imputer.transform(X_predict)
X_predict_selected = selector.transform(X_predict_imputed)

# Hacer predicciones con el modelo
predictions = best_model.predict(X_predict_selected)

# Agregar una columna 'SalePrice' al DataFrame df_predict con las predicciones
df_predict['SalePrice'] = np.exp(predictions)

# Guardar el DataFrame como un archivo CSV
output_path = 'data/predictions/data_predicted.csv'
save_dataframe(df_predict, output_path)

print('Se ha generado un archivo con la predicción de precios de casas, puede proceder a descargarlo de la carpeta data/predictions.')