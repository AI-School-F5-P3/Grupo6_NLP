from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC  # Importa el modelo SVM
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
import os
from joblib import dump

def load_data(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', file_name)
    return pd.read_csv(data_path)

def calculate_overfitting(model, X_train, X_test, y_train, y_test):
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    overfitting = (train_accuracy - test_accuracy) / train_accuracy * 100
    return train_accuracy, test_accuracy, overfitting

def train_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"{model_name} Results:")
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
    
    # Calculate training accuracy and overfitting metrics
    train_accuracy, test_accuracy, overfitting = calculate_overfitting(model, X_train, X_test, y_train, y_test)
    
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Overfitting: {overfitting:.2f}%")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'src', 'metrics', f'{model_name}_confusion_matrix.png')
    plt.savefig(file_path)
    plt.close()
    
    return model

def train_svm():
    # Load preprocessed data
    df = load_data('model_df.csv')

    # Ensure 'IsToxic' is binary 
    df['IsToxic'] = df['IsToxic'].astype(int)
    
    # Vectorización con TF-IDF
    vectorizer = TfidfVectorizer(max_features = 500, min_df=5, max_df=0.7)
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['IsToxic']  
    
    # Dividir el dataset en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Crear y configurar el modelo SVM
    model = SVC(kernel='rbf', C=0.3, probability=True,  # Necesario para predict_proba
        random_state=42
    )  
    
   
    # Evaluar el modelo
    model = train_evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        "SVM Classifier"
    )
    # Validación cruzada para evaluar el modelo y reducir overfitting
    cv_scores = cross_val_score(model, X, y, cv=5)
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Promedio de Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%")
    
    # Evaluar el modelo
    model = train_evaluate_model(
        model,
        X_train,
        X_test,
        y_train,
        y_test,
        "SVM Classifier"
    )

    print("SVM training completed and saved.")
    return model

if __name__ == "__main__":
    train_svm()