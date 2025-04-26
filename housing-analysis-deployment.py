"""
ANÁLISIS DE PRECIOS DE VIVIENDA Y MODELO PREDICTIVO
Este ejemplo muestra el proceso completo de análisis de datos y despliegue
"""

# PASO 1: Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# PASO 2: Cargar y examinar los datos
# Usamos el conjunto de datos de viviendas de Boston como ejemplo
from sklearn.datasets import load_boston

# Nota: En una aplicación real, cargaríamos datos desde archivos o bases de datos
# boston = load_boston()
# df = pd.DataFrame(boston.data, columns=boston.feature_names)
# df['PRICE'] = boston.target

# Como el dataset de Boston está obsoleto, creamos un dataset sintético similar
np.random.seed(42)
n_samples = 500

# Crear características sintéticas
RM = np.random.normal(6.5, 1, n_samples)  # Número medio de habitaciones
LSTAT = np.random.normal(12, 7, n_samples)  # % de población de estatus bajo
PTRATIO = np.random.normal(18, 2, n_samples)  # Ratio alumno-profesor
DIS = np.random.normal(3.5, 2, n_samples)  # Distancia ponderada a centros de empleo

# Crear precio como función de estas características más algo de ruido
PRICE = 22 + 5*RM - 0.7*LSTAT - 0.7*PTRATIO - 0.4*DIS + np.random.normal(0, 3, n_samples)

# Crear DataFrame
data = {
    'RM': RM,
    'LSTAT': LSTAT,
    'PTRATIO': PTRATIO,
    'DIS': DIS,
    'PRICE': PRICE
}
df = pd.DataFrame(data)

# Guardar el DataFrame para usarlo en la aplicación Streamlit
df.to_csv('housing_data.csv', index=False)

# PASO 3: Análisis exploratorio de datos
print("Primeras 5 filas del DataFrame:")
print(df.head())

print("\nEstadísticas descriptivas:")
print(df.describe())

# Visualizaciones
plt.figure(figsize=(12, 8))

# Matriz de correlación
plt.subplot(2, 2, 1)
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matriz de Correlación')

# Relación entre RM y PRICE
plt.subplot(2, 2, 2)
plt.scatter(df['RM'], df['PRICE'])
plt.xlabel('Número medio de habitaciones (RM)')
plt.ylabel('Precio (PRICE)')
plt.title('Relación entre RM y PRICE')

# Relación entre LSTAT y PRICE
plt.subplot(2, 2, 3)
plt.scatter(df['LSTAT'], df['PRICE'])
plt.xlabel('% de población de estatus bajo (LSTAT)')
plt.ylabel('Precio (PRICE)')
plt.title('Relación entre LSTAT y PRICE')

# Distribución de PRICE
plt.subplot(2, 2, 4)
sns.histplot(df['PRICE'], kde=True)
plt.xlabel('Precio (PRICE)')
plt.title('Distribución de PRICE')

plt.tight_layout()
plt.savefig('eda_visualizations.png')

# PASO 4: Preparación de los datos para el modelado
X = df.drop('PRICE', axis=1)
y = df['PRICE']

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PASO 5: Modelado
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Evaluación del modelo
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nResultados del modelo:")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Coeficientes del modelo
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print("\nCoeficientes del modelo:")
print(coefficients)

# PASO 6: Guardar el modelo y el scaler para su despliegue
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/housing_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("\nModelo y scaler guardados correctamente.")
