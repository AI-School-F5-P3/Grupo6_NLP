from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, classification_report, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import learning_curve
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
    
    train_accuracy, test_accuracy, overfitting = calculate_overfitting(
        model, X_train, X_test, y_train, y_test
    )
    
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Overfitting: {overfitting:.2f}%")
    
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    
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
    
    # Create a pipeline with much stricter feature selection and regularization
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=150,          # Reducido significativamente
            min_df=20,                 # Aumentado para eliminar términos raros
            max_df=0.3,               # Reducido para eliminar términos muy comunes
            ngram_range=(1, 1),        # Solo unigramas para reducir complejidad
            stop_words='english'       # Eliminar palabras comunes
        )),
        ('feature_selection', SelectKBest(chi2, k=100)),  # Reducido número de features
        ('scaler', StandardScaler(with_mean=False)),
        ('svm', SVC(probability=True, 
                   random_state=42,
                   kernel='linear',     # Kernel lineal por defecto
                   C=0.1,              # Valor bajo de C para mayor regularización
                   class_weight='balanced'))
    ])
    
    # Split data with larger test size
    X = df['processed_text']
    y = df['IsToxic']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # Reduced parameter grid focusing on regularization
    param_grid = {
        'svm__C': [0.01, 0.05, 0.1],           # Valores más bajos de C
        'tfidf__max_features': [100, 150, 200], # Probar diferentes cantidades de features
        'feature_selection__k': [50, 100],      # Probar diferentes números de features seleccionadas
    }
    
    # Cross validation con más folds
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # Grid search con early stopping
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='balanced_accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Plot learning curves
    #plot_learning_curves(best_model, X, y, cv)
    
    # Evaluate best model
    model = train_evaluate_model(
        best_model,
        X_train,
        X_test,
        y_train,
        y_test,
        "SVM Classifier"
    )
    

    # Save model and vectorizer
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models')
    dump(model, os.path.join(model_path, 'svm.joblib'))
    
    print("SVM completed and saved.")
    return model

if __name__ == "__main__":
    train_svm()