import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import os
from joblib import dump
from wordcloud import WordCloud

def load_data(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', file_name)
    return pd.read_csv(data_path)

def calculate_overfitting(model, X_train, X_test, y_train, y_test):
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    overfitting = (train_accuracy - test_accuracy) / train_accuracy * 100
    return train_accuracy, test_accuracy, overfitting

def train_evaluate_model(pipeline, X_train, X_test, y_train, y_test, model_name):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    
    print(f"{model_name} Results:")
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.4f}")
    
    # Calculate training accuracy and overfitting metrics
    train_accuracy, test_accuracy, overfitting = calculate_overfitting(pipeline, X_train, X_test, y_train, y_test)
    
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
    
    return pipeline

def create_wordcloud(df):
    toxic_text = ' '.join(df[df['IsToxic'] == 1]['processed_text'])
    
    wordcloud = WordCloud(width=800, height=400,
                        background_color='white').generate(toxic_text)
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Most Common Words in Toxic Comments')
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'src', 'metrics')
    plt.savefig(file_path)
    plt.close()

def train_logistic_regresion():
    # Load preprocessed data
    df = load_data('model_df.csv')

    # Ensure 'IsToxic' is binary 
    df['IsToxic'] = df['IsToxic'].astype(int)
    
    # Create pipeline with PCA for reducing overfitting
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('pca', PCA(n_components=100)),
        ('clf', LogisticRegression(random_state=42, max_iter=2000))
    ])
    
    # Define grid of hyperparameters
    param_grid = {
        'tfidf__max_features': [800, 1000, 1200],
        'tfidf__ngram_range': [(1,1), (1,2)],
        'tfidf__min_df': [2, 3, 4],
        'pca__n_components': [100, 150, 200],
        'clf__C': [0.5, 1.0, 2.0],
        'clf__penalty': ['l2'],
        'clf__class_weight': ['balanced'],
    }
    
    # Create and run GridSearchCV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )

    # Fit the model
    print("Starting hyperparameter search...")
    grid_search.fit(df['processed_text'], df['IsToxic'])
    
    # Get and display the best parameters
    print("\nBest parameters found:")
    print(grid_search.best_params_)

    # Create final model with the best parameters
    best_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=grid_search.best_params_['tfidf__max_features'],
            ngram_range=grid_search.best_params_['tfidf__ngram_range'],
            min_df=grid_search.best_params_['tfidf__min_df']
        )),
        ('pca', PCA(n_components=grid_search.best_params_['pca__n_components'])),
        ('clf', LogisticRegression(
            C=grid_search.best_params_['clf__C'],
            class_weight=grid_search.best_params_['clf__class_weight'],
            penalty=grid_search.best_params_['clf__penalty'],
            random_state=42,
            max_iter=2000
        ))
    ])
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(df['processed_text'], df['IsToxic'], test_size=0.2, random_state=42)

    # Train and evaluate the model
    best_pipeline = train_evaluate_model(best_pipeline,
                                         X_train, X_test,
                                         y_train, y_test,
                                         "Logistic Regression with PCA")
    
    # Save model and vectorizer
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models')
    dump(best_pipeline, os.path.join(model_path, 'logistic_regression_model_with_pca.joblib'))
    
    print("Logistic Regression model with PCA training completed and saved.")
    return best_pipeline

if __name__ == "__main__":
    train_logistic_regresion()