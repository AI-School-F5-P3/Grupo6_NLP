# YouTube Toxicity Detector 🕵️‍♀️

Este proyecto desarrolla e implementa una solución automatizada y escalable para detectar y clasificar mensajes de odio en comentarios de YouTube. El objetivo es facilitar la moderación de contenido y reducir la carga de trabajo de los moderadores humanos, priorizando la practicidad y capacidad de procesamiento sobre la precisión absoluta.

## 🚀 Objetivo
Implementar una herramienta que permita:
- **Automatización:** Clasificar contenido tóxico sin intervención humana.
- **Escalabilidad:** Manejar grandes volúmenes de datos.
- **Detección efectiva:** Identificar mensajes de odio con precisión razonable.
- **Eficiencia:** Priorizar velocidad y capacidad de implementación sobre perfección absoluta.
- **Acciones inmediatas:** Permitir a los moderadores eliminar mensajes o restringir usuarios rápidamente.


## 📂 Estructura del Proyecto
<img width="429" alt="image" src="https://github.com/user-attachments/assets/b73650bb-4883-448a-8f7d-5268d8dc2c0f">


## 🔧 Funcionalidades

### Procesamiento de Lenguaje Natural (NLP)
- Tokenización, eliminación de stopwords, lematización o stemming.
- Vectorización con **TF-IDF** y embeddings (Word2Vec y BERT).
- Análisis de emojis para clasificar contenido potencialmente tóxico.

### Modelos Implementados
- **Complement Naive Bayes:** Ideal para clasificación binaria.
- **Regresión Logística:** Usando PCA para reducción de dimensionalidad.
- **Random Forest:** Combinación de TF-IDF y Word2Vec.
- **Ensamblaje:** Clasificadores Stacking y Voting integrando SVM y regresión logística.

### Métricas y Evaluación
- Matrices de confusión.
- Visualización de importancia de características y nubes de palabras.
- Métricas como ROC-AUC, F1-Score y Balanced Accuracy.

### Interfaz Gráfica
- Desarrollada en Streamlit.
- Funcionalidades:
  - Análisis de comentarios individuales.
  - Métricas y visualizaciones del rendimiento del modelo.
  - Análisis automatizado de comentarios desde un video de YouTube.

---

## 🛠️ Instalación y Configuración

### Requisitos previos 
- Python 3.8 o superior.
- Dependencias listadas en `requirements.txt`.
- Clave de API de YouTube para analizar comentarios.

### Pasos de Instalación
1. Clonar el repositorio:
   ```bash
   git clone https://github.com/AI-School-F5-P3/Grupo6_NLP.git
  
2. Crear y activar un entorno virtual: 
   ```bash
    python -m venv venv
    source venv/bin/activate # En Windows: venv\Scripts\activate

3. Instalar dependencias:
   ```bash
    pip install -r requirements.txt

4. Configurar las variables de entorno:
    - Crear un archivo .env en el directorio raíz.
    - Añadir la clave de API de YouTube:
       ```bash
        YOUTUBE_API_KEY=tu_api_key

5. Ejecutar la aplicación:
   ```bash
    python run_app.py

## 🤝 Contribuciones
¡Contribuciones son bienvenidas! Por favor, abre un Issue o envía un Pull Request para colaborar.

## 📜 Licencia
Este proyecto está licencia MIT.
