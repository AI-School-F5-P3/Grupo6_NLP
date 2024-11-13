import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from wordcloud import WordCloud
import optuna

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

def create_wordcloud(df):
    toxic_text = ' '.join(df[df['IsToxic'] == 1]['processed_text'])
    
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white').generate(toxic_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in Toxic Comments')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'src', 'metrics', 'toxic_wordcloud.png')
    plt.savefig(file_path)
    plt.close()

def train_naive_bayes():
    # Load preprocessed data
    df = load_data('model_df.csv')

    # Ensure 'IsToxic' is binary 
    df['IsToxic'] = df['IsToxic'].astype(int)
    
    # Create a word cloud for toxic comments before vectorization
    create_wordcloud(df)
    
    # Reduce the number of features
    vectorizer = TfidfVectorizer(max_features=500, min_df=5, max_df=0.7)
    X_reduced = vectorizer.fit_transform(df['processed_text'])
    y = df['IsToxic']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)
    
    # Apply TruncatedSVD for dimensionality reduction
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_train_reduced = svd.fit_transform(X_train)
    X_test_reduced = svd.transform(X_test)
    
    # Convert to absolute values
    X_train_reduced = abs(X_train_reduced)
    X_test_reduced = abs(X_test_reduced)

    # Optuna hyperparameter optimization
    def objective(trial):
        alpha = trial.suggest_float('alpha', 0.5, 2.0)
        norm = trial.suggest_categorical('norm', [True, False])
        model = ComplementNB(alpha=alpha, norm=norm)
        
        skf = StratifiedKFold(n_splits=5)
        scores = cross_val_score(model, X_train_reduced, y_train, cv=skf, scoring='roc_auc')
        return scores.mean()
    
    # Run Optuna optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    best_alpha = study.best_params['alpha']
    best_norm = study.best_params['norm']
    print(f"Best alpha found by Optuna: {best_alpha}")
    print(f"Best norm found by Optuna: {best_norm}")
    
    # Train final model with the best parameters
    best_model = ComplementNB(alpha=best_alpha, norm=best_norm)
    best_model.fit(X_train_reduced, y_train)
    
    # Evaluate the model
    best_model = train_evaluate_model(best_model, X_train_reduced, X_test_reduced, y_train, y_test, "Complement_Naive_Bayes_SVD")
    
    # Perform cross-validation
    cv_scores = cross_val_score(best_model, X_reduced, y, cv=5)
    print(f"Cross-Validation Accuracy Scores: {cv_scores}")
    print(f"Average Cross-Validation Accuracy: {cv_scores.mean() * 100:.2f}%")
    
    # Feature Importance Analysis
    feature_importance = best_model.feature_log_prob_[1] - best_model.feature_log_prob_[0]
    importance_df = pd.DataFrame({
        'feature': [f'component_{i+1}' for i in range(svd.n_components)],
        'importance': feature_importance
    })
    importance_df = importance_df.sort_values('importance', ascending=False).head(20)
    
    # Plot top 20 important features
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_df)
    plt.title('Top 20 Most Important Features')
    plt.tight_layout()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'src', 'metrics', 'feature_importance.png')
    plt.savefig(file_path)
    plt.close()

    # Save model and vectorizer
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models')
    dump(best_model, os.path.join(model_path, 'complement_naive_bayes_model.joblib'))
    dump(vectorizer, os.path.join(model_path, 'tfidf_vectorizer.joblib'))
    
    print("Complement Naive Bayes model training completed and saved.")
    return best_model

if __name__ == "__main__":
    train_naive_bayes()
