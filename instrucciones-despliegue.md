# Instrucciones para el Despliegue

Este documento explica cómo desplegar la aplicación de análisis de datos y predicción de precios de viviendas utilizando diferentes opciones gratuitas.

## Opción 1: Streamlit Sharing (Recomendada)

Streamlit Sharing es una plataforma gratuita que permite desplegar aplicaciones Streamlit de forma sencilla.

### Requisitos previos
- Una cuenta de GitHub
- Una cuenta en [Streamlit Sharing](https://streamlit.io/sharing)

### Pasos para el despliegue

1. **Crear un repositorio en GitHub**
   - Sube todos los archivos del proyecto a un nuevo repositorio en GitHub
   - Asegúrate de incluir los siguientes archivos:
     - `app.py` (la aplicación Streamlit)
     - `housing_data.csv` (o el archivo de datos que uses)
     - `/models/housing_model.pkl` y `/models/scaler.pkl`
     - `requirements.txt`
     - `.streamlit/config.toml` (opcional)

2. **Desplegar en Streamlit Sharing**
   - Inicia sesión en [Streamlit Sharing](https://streamlit.io/sharing)
   - Haz clic en "New app"
   - Selecciona tu repositorio de GitHub
   - En "Main file path", escribe `app.py`
   - Haz clic en "Deploy"

3. **Verificar el despliegue**
   - Una vez completado, se te proporcionará una URL para acceder a tu aplicación
   - Comprueba que la aplicación funciona correctamente

## Opción 2: Despliegue Local con Docker

Docker permite empaquetar la aplicación y todas sus dependencias en un contenedor que se puede ejecutar en cualquier entorno.

### Requisitos previos
- [Docker](https://www.docker.com/get-started) instalado en tu sistema

### Pasos para el despliegue

1. **Construir la imagen de Docker**
   ```bash
   docker build -t housing-app .
   ```

2. **Ejecutar el contenedor**
   ```bash
   docker run -p 8501:8501 housing-app
   ```

3. **Acceder a la aplicación**
   - Abre un navegador y navega a `http://localhost:8501`

## Opción 3: Despliegue en Heroku

Heroku ofrece una capa gratuita para aplicaciones web que es adecuada para este tipo de proyectos.

### Requisitos previos
- [Git](https://git-scm.com/) instalado
- Una cuenta en [Heroku](https://www.heroku.com/)
- [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) instalado

### Pasos para el despliegue

1. **Inicializar un repositorio Git (si no está ya inicializado)**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   ```

2. **Crear una aplicación en Heroku**
   ```bash
   heroku login
   heroku create nombre-de-tu-app
   ```

3. **Desplegar la aplicación**
   ```bash
   git push heroku main
   ```

4. **Verificar el despliegue**
   ```bash
   heroku open
   ```

## Opción 4: Despliegue en PythonAnywhere

PythonAnywhere ofrece una opción gratuita para alojar aplicaciones web Python.

### Requisitos previos
- Una cuenta en [PythonAnywhere](https://www.pythonanywhere.com/)

### Pasos para el despliegue

1. **Crear una cuenta y acceder a PythonAnywhere**

2. **Configurar un nuevo entorno virtual**
   - Ve a la pestaña "Consoles" y abre una consola Bash
   - Crea un entorno virtual:
     ```bash
     mkvirtualenv --python=python3.9 myenv
     ```

3. **Clonar el repositorio o subir los archivos**
   - Si tienes un repositorio Git:
     ```bash
     git clone https://github.com/tu-usuario/tu-repositorio.git
     ```
   - Alternativamente, puedes subir los archivos utilizando la interfaz web

4. **Instalar dependencias**
   ```bash
   cd tu-repositorio
   pip install -r requirements.txt
   ```

5. **Configurar una aplicación web**
   - Ve a la pestaña "Web"
   - Añade una nueva aplicación web
   - Selecciona "Manual Configuration" y Python 3.9
   - Configura la ruta al archivo WSGI para ejecutar tu aplicación Streamlit
   - Guarda los cambios y recarga la aplicación

## Mantenimiento y Monitoreo

Independientemente de la opción de despliegue elegida, es importante considerar:

1. **Monitoreo**
   - Verifica periódicamente que la aplicación funcione correctamente
   - Revisa si hay errores en los logs

2. **Actualizaciones**
   - Actualiza el modelo periódicamente si los datos cambian
   - Mantén las dependencias actualizadas para evitar problemas de seguridad

3. **Escalabilidad**
   - Si la aplicación comienza a recibir más tráfico, considera actualizar a un plan de pago o migrar a una solución más robusta

## Recursos Adicionales

- [Documentación de Streamlit](https://docs.streamlit.io/)
- [Guía de despliegue de Streamlit](https://docs.streamlit.io/streamlit-cloud/get-started)
- [Documentación de Docker](https://docs.docker.com/)
- [Documentación de Heroku para Python](https://devcenter.heroku.com/categories/python-support)
- [Guía de PythonAnywhere](https://help.pythonanywhere.com/pages/)
