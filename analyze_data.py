import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

# Crear un dataset sintético de viviendas
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

# Guardar el CSV
df.to_csv('housing_data.csv', index=False)
print("Dataset creado y guardado en housing_data.csv")

# Crear y guardar el modelo
X = df.drop('PRICE', axis=1)
y = df['PRICE']

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Estandarizar las características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Entrenar el modelo
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Crear directorio para los modelos si no existe
os.makedirs('models', exist_ok=True)

# Guardar el modelo y el scaler
joblib.dump(model, 'models/housing_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("Modelo y scaler creados y guardados en la carpeta models/")

# Mostrar las primeras filas del dataset
print("\nPrimeras 5 filas del DataFrame:")
print(df.head())

# Mostrar estadísticas básicas
print("\nEstadísticas descriptivas:")
print(df.describe())

# Ver los coeficientes del modelo
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})
print("\nCoeficientes del modelo:")
print(coefficients)