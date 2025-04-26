"""
APLICACIÓN STREAMLIT PARA DESPLIEGUE DEL MODELO DE PRECIOS DE VIVIENDAS
Esta aplicación permite a los usuarios interactuar con el modelo para predecir precios de viviendas
y visualizar el análisis exploratorio de datos.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from PIL import Image

# Configurar la página
st.set_page_config(
    page_title="Predictor de Precios de Viviendas",
    page_icon="🏠",
    layout="wide"
)

# Funciones para cargar datos y modelos
@st.cache_data
def load_data():
    try:
        return pd.read_csv('housing_data.csv')
    except FileNotFoundError:
        # Datos de ejemplo si no se encuentra el archivo
        data = {
            'RM': np.linspace(4, 9, 500),
            'LSTAT': np.linspace(1, 40, 500),
            'PTRATIO': np.linspace(12, 22, 500),
            'DIS': np.linspace(1, 10, 500),
            'PRICE': np.linspace(5, 50, 500)
        }
        return pd.DataFrame(data)

@st.cache_resource
def load_model():
    try:
        model = joblib.load('models/housing_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("No se encontraron los archivos del modelo. Ejecute primero el análisis.")
        return None, None

# Cargar datos y modelo
df = load_data()
model, scaler = load_model()

# Título de la aplicación
st.title("🏠 Predictor de Precios de Viviendas")
st.markdown("Esta aplicación permite predecir el precio de viviendas basado en características clave.")

# Sidebar para navegación
page = st.sidebar.radio("Navegación", ["Inicio", "Análisis Exploratorio", "Predicción", "Acerca de"])

# Página de inicio
if page == "Inicio":
    st.header("Bienvenido al Predictor de Precios de Viviendas")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 📊 ¿Qué puede hacer esta aplicación?
        
        - **Explorar datos** de viviendas y sus características
        - **Visualizar relaciones** entre diferentes variables
        - **Predecir precios** basados en un modelo entrenado
        
        Utilice el menú de navegación para explorar las diferentes secciones.
        """)
        
        st.subheader("Vista previa de los datos")
        st.dataframe(df.head())
    
    with col2:
        st.markdown("### 📈 Precio promedio por número de habitaciones")
        # Gráfico simple para la página de inicio
        fig, ax = plt.subplots()
        df.groupby(pd.cut(df['RM'], bins=5)).mean()['PRICE'].plot(kind='bar', ax=ax)
        ax.set_xlabel('Número de habitaciones (agrupado)')
        ax.set_ylabel('Precio promedio')
        st.pyplot(fig)

# Página de análisis exploratorio
elif page == "Análisis Exploratorio":
    st.header("Análisis Exploratorio de Datos")
    
    st.markdown("""
    Esta sección muestra diferentes visualizaciones de los datos para entender mejor las relaciones
    entre las variables y su impacto en el precio de las viviendas.
    """)
    
    # Mostrar visualizaciones
    try:
        image = Image.open('eda_visualizations.png')
        st.image(image, caption='Visualizaciones del análisis exploratorio de datos')
    except FileNotFoundError:
        st.warning("La imagen de visualización no está disponible. Generando visualizaciones...")
        
        # Código para generar visualizaciones si no existen
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Matriz de correlación
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[0, 0])
        axes[0, 0].set_title('Matriz de Correlación')
        
        # RM vs PRICE
        axes[0, 1].scatter(df['RM'], df['PRICE'])
        axes[0, 1].set_xlabel('Número medio de habitaciones (RM)')
        axes[0, 1].set_ylabel('Precio (PRICE)')
        axes[0, 1].set_title('Relación entre RM y PRICE')
        
        # LSTAT vs PRICE
        axes[1, 0].scatter(df['LSTAT'], df['PRICE'])
        axes[1, 0].set_xlabel('% de población de estatus bajo (LSTAT)')
        axes[1, 0].set_ylabel('Precio (PRICE)')
        axes[1, 0].set_title('Relación entre LSTAT y PRICE')
        
        # Distribución de PRICE
        sns.histplot(df['PRICE'], kde=True, ax=axes[1, 1])
        axes[1, 1].set_xlabel('Precio (PRICE)')
        axes[1, 1].set_title('Distribución de PRICE')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Análisis adicionales interactivos
    st.subheader("Exploración Interactiva")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_var = st.selectbox("Variable X", options=df.columns.tolist())
    
    with col2:
        y_var = st.selectbox("Variable Y", options=df.columns.tolist(), index=4)  # Default to PRICE
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_var, y=y_var, ax=ax)
    ax.set_title(f'Relación entre {x_var} y {y_var}')
    st.pyplot(fig)
    
    # Estadísticas descriptivas
    st.subheader("Estadísticas Descriptivas")
    st.dataframe(df.describe())

# Página de predicción
elif page == "Predicción":
    st.header("Predicción de Precios de Viviendas")
    
    st.markdown("""
    En esta sección puede ingresar las características de una vivienda y obtener una predicción 
    del precio basada en el modelo entrenado.
    """)
    
    if model is not None and scaler is not None:
        # Formulario para ingresar valores
        with st.form("prediction_form"):
            st.subheader("Ingrese las características de la vivienda")
            
            col1, col2 = st.columns(2)
            
            with col1:
                rm = st.slider("Número medio de habitaciones (RM)", 
                               float(df['RM'].min()), 
                               float(df['RM'].max()), 
                               float(df['RM'].mean()))
                
                lstat = st.slider("% de población de estatus bajo (LSTAT)", 
                                  float(df['LSTAT'].min()), 
                                  float(df['LSTAT'].max()), 
                                  float(df['LSTAT'].mean()))
            
            with col2:
                ptratio = st.slider("Ratio alumno-profesor (PTRATIO)", 
                                    float(df['PTRATIO'].min()), 
                                    float(df['PTRATIO'].max()), 
                                    float(df['PTRATIO'].mean()))
                
                dis = st.slider("Distancia a centros de empleo (DIS)", 
                                float(df['DIS'].min()), 
                                float(df['DIS'].max()), 
                                float(df['DIS'].mean()))
            
            submit_button = st.form_submit_button("Predecir Precio")
        
        # Mostrar predicción cuando se envía el formulario
        if submit_button:
            # Crear un array con los valores ingresados
            input_data = np.array([[rm, lstat, ptratio, dis]])
            
            # Estandarizar los datos
            input_scaled = scaler.transform(input_data)
            
            # Realizar la predicción
            prediction = model.predict(input_scaled)[0]
            
            # Mostrar el resultado
            st.success(f"El precio predicho para esta vivienda es: ${prediction:.2f}k")
            
            # Mostrar interpretación de la predicción
            st.subheader("Interpretación de la predicción")
            
            # Contribución de cada variable a la predicción
            coefficients = model.coef_
            feature_names = ['RM', 'LSTAT', 'PTRATIO', 'DIS']
            
            # Calcular el impacto de cada variable en la predicción
            scaled_values = input_scaled[0]
            impacts = coefficients * scaled_values
            
            # Crear un DataFrame para mostrar la contribución
            contribution_df = pd.DataFrame({
                'Variable': feature_names,
                'Valor': input_data[0],
                'Coeficiente': coefficients,
                'Impacto': impacts
            })
            
            # Ordenar por impacto absoluto
            contribution_df['Impacto Abs'] = contribution_df['Impacto'].abs()
            contribution_df = contribution_df.sort_values('Impacto Abs', ascending=False)
            
            # Mostrar gráfico de barras con el impacto de cada variable
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(contribution_df['Variable'], contribution_df['Impacto'])
            
            # Colorear barras según impacto positivo o negativo
            for i, bar in enumerate(bars):
                if contribution_df['Impacto'].iloc[i] < 0:
                    bar.set_color('red')
                else:
                    bar.set_color('green')
                    
            ax.set_xlabel('Impacto en la predicción')
            ax.set_title('Contribución de cada variable a la predicción')
            st.pyplot(fig)
            
            # Mostrar tabla de contribución
            st.dataframe(contribution_df[['Variable', 'Valor', 'Coeficiente', 'Impacto']])
    else:
        st.error("No se pudo cargar el modelo. Verifique si los archivos del modelo existen.")

# Página acerca de
elif page == "Acerca de":
    st.header("Acerca de esta aplicación")
    
    st.markdown("""
    ### Proyecto de Despliegue en Ciencia de Datos
    
    Esta aplicación fue creada como ejemplo para un curso de despliegue en ciencia de datos. 
    Demuestra cómo desplegar un modelo de predicción de precios de viviendas utilizando 
    herramientas gratuitas como Streamlit.
    
    #### Herramientas utilizadas:
    - **Python**: Lenguaje de programación principal
    - **Pandas & NumPy**: Manipulación y análisis de datos
    - **Scikit-learn**: Construcción y evaluación del modelo
    - **Matplotlib & Seaborn**: Visualización de datos
    - **Streamlit**: Framework para crear la aplicación web
    - **Joblib**: Serialización del modelo
    
    #### Despliegue:
    Esta aplicación puede desplegarse gratuitamente utilizando Streamlit Sharing o servicios similares.
    
    #### Código fuente:
    El código completo está disponible en el repositorio del curso.
    """)
