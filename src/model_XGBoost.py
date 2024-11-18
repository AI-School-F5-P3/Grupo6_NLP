import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
from wordcloud import WordCloud
import optuna
import xgboost as xgb
from gensim.models import KeyedVectors
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

# Load GloVe embeddings
def load_glove_model(glove_file):
    print("Loading GloVe Model...")
    glove_model = KeyedVectors.load_word2vec_format(glove_file, binary=False, no_header=True)
    print("GloVe Model Loaded!")
    return glove_model

# Vectorize text using GloVe
def vectorize_text(text, glove_model, dim=100):
    tokens = word_tokenize(text.lower())
    vec = np.zeros(dim)
    count = 0
    for token in tokens:
        if token in glove_model:
            vec += glove_model[token]
            count += 1
    if count > 0:
        vec /= count
    return vec

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
    
    train_accuracy, test_accuracy, overfitting = calculate_overfitting(model, X_train, X_test, y_train, y_test)
    
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

def create_wordcloud(df):
    toxic_text = ' '.join(df[df['IsToxic'] == 1]['processed_text'])
    
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(toxic_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in Toxic Comments')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'src', 'metrics', 'toxic_wordcloud.png')
    plt.savefig(file_path)
    plt.close()

def predict_toxicity(text, model, glove_model, svd):
    vec = vectorize_text(text, glove_model)
    vec_reshaped = vec.reshape(1, -1)
    vec_reduced = svd.transform(vec_reshaped)
    prediction = model.predict(vec_reduced)
    probability = model.predict_proba(vec_reduced)[0][1]
    return prediction[0], probability

def train_xgboost():
    # Load preprocessed data
    df = load_data('model_df.csv')
    df['IsToxic'] = df['IsToxic'].astype(int)
    
    create_wordcloud(df)
    
    # Load GloVe embeddings
    glove_file = 'path_to_your_glove_file/glove.twitter.27B.100d.txt'  # Update this path
    glove_model = load_glove_model(glove_file)
    
    # Vectorize text using GloVe
    X = np.array([vectorize_text(text, glove_model) for text in df['processed_text']])
    y = df['IsToxic'].values
    
    # Apply TruncatedSVD for dimensionality reduction
    svd = TruncatedSVD(n_components=50, random_state=42)
    X_reduced = svd.fit_transform(X)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
    
    # Optuna hyperparameter optimization
    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 1.0),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
        }
        model = xgb.XGBClassifier(**params, random_state=42)
        
        skf = StratifiedKFold(n_splits=5)
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='roc_auc')
        return scores.mean()
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)
    
    best_params = study.best_params
    print("Best hyperparameters:", best_params)
    
    # Train final model with best parameters
    best_model = xgb.XGBClassifier(**best_params, random_state=42)
    best_model = train_evaluate_model(best_model, X_train, X_test, y_train, y_test, "XGBoost_GloVe")
    
    # Cross-validation
    cv_scores = cross_val_score(best_model, X_reduced, y, cv=5)
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Average Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%")
    
    # Feature Importance
    feature_importance = best_model.feature_importances_
    feature_names = [f"feature_{i}" for i in range(X_reduced.shape[1])]
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    })
    importance_df = importance_df.sort_values('importance', ascending=False).head(20)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'src', 'metrics', 'feature_importance_xgboost.png')
    plt.savefig(file_path)
    plt.close()

    # Save model, GloVe model, and SVD
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models')
    dump(best_model, os.path.join(model_path, 'xgboost_model.joblib'))
    dump(glove_model, os.path.join(model_path, 'glove_model.joblib'))
    dump(svd, os.path.join(model_path, 'svd_model_xgboost.joblib'))

    print("XGBoost model, GloVe model, and SVD model training completed and saved.")
    return best_model, glove_model, svd

if __name__ == "__main__":
    train_xgboost()