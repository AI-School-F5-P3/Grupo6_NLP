import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.models import Word2Vec
from joblib import dump
from wordcloud import WordCloud

def load_data(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', file_name)
    return pd.read_csv(data_path)

def train_word2vec(texts, vector_size=100, window=5, min_count=1):
    sentences = [text.split() for text in texts]
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=4)
    return model

def text_to_vec(text, model):
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv.key_to_index]
    if len(word_vectors) == 0:
        return np.zeros(model.vector_size)
    return np.mean(word_vectors, axis=0)

def transform_text_to_vec(texts, model):
    return np.array([text_to_vec(text, model) for text in texts])

def calculate_overfitting(model, X_train, X_test, y_train, y_test):
    train_accuracy = model.score(X_train, y_train)
    test_accuracy = model.score(X_test, y_test)
    overfitting = (train_accuracy - test_accuracy) / train_accuracy * 100
    return train_accuracy, test_accuracy, overfitting

def train_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    # Fit the model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    print(f"{model_name} Results:")
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
    
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
    plt.savefig(f'{model_name}_confusion_matrix.png')
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

def train_random_forest():
   # Load preprocessed data
   df = load_data('model_df.csv')
   
   # Create a word cloud for toxic comments before vectorization
   create_wordcloud(df)

   # Vectorization using TF-IDF
   vectorizer = TfidfVectorizer(max_features=5000)
   X_tfidf = vectorizer.fit_transform(df['processed_text']).toarray()
   
   # Train Word2Vec model and create word embeddings
   word2vec_model = train_word2vec(df['processed_text'])
   X_word2vec = transform_text_to_vec(df['processed_text'], word2vec_model)
   
   # Combine TF-IDF and Word2Vec features
   X_combined = np.hstack((X_tfidf, X_word2vec))
   y = df['IsToxic']
   
   # Split the data
   X_train, X_test, y_train, y_test = train_test_split(X_combined, y,
                                                       test_size=0.2,
                                                       random_state=42)
   
   # Compute class weights for imbalanced data
   class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
   class_weight_dict = dict(zip(np.unique(y), class_weights))
   
   # Define Random Forest model and parameters for GridSearch
   model = RandomForestClassifier(random_state=42)

   param_grid = {
       'n_estimators': [100],
       'max_depth': [10],
       'min_samples_split': [2],
       'class_weight': [class_weight_dict]
   }
   
   grid_search = GridSearchCV(model,
                              param_grid=param_grid,
                              cv=5,
                              scoring='roc_auc',
                              n_jobs=-1)
   
   grid_search.fit(X_train,y_train)
   
   best_model = grid_search.best_estimator_
   
   print(f"Best parameters for Random Forest: {grid_search.best_params_}")
   
   # Evaluate model with metrics output to console
   best_model = train_evaluate_model(best_model,
                                      X_train,
                                      X_test,
                                      y_train,
                                      y_test,
                                      "Random_Forest")
   
   # Feature Importance Analysis (only for Random Forest)
   if isinstance(best_model, RandomForestClassifier):
       feature_importance = best_model.feature_importances_
       # Use only TF-IDF features
       feature_names = vectorizer.get_feature_names_out()

       # Ensure feature_importance matches the number of TF-IDF features
       feature_importance = feature_importance[:len(feature_names)]
       importance_df = pd.DataFrame({'feature': feature_names,
                                      'importance': feature_importance})
       importance_df = importance_df.sort_values('importance', ascending=False).head(20)

       plt.figure(figsize=(12, 8))
       sns.barplot(x='importance', y='feature', data=importance_df)
       plt.title('Top 20 Most Important Features')
       plt.tight_layout()
       
       # Create the file path
       current_dir = os.path.dirname(os.path.abspath(__file__))
       file_path = os.path.join(current_dir, '..', 'src', 'metrics','feature_importance.png')

       # Save the figure
       plt.savefig(file_path)
       plt.close()
   
   # Create the model path
   current_dir = os.path.dirname(os.path.abspath(__file__))
   model_path = os.path.join(current_dir, 'models')

   # Save the best model and vectorizer
   dump(best_model,os.path.join(model_path, 'random_forest_model.joblib'))
   dump(vectorizer,os.path.join(model_path,'tfidf_vectorizer.joblib'))
   dump(word2vec_model,os.path.join(model_path,'word2vec_model.joblib'))

if __name__ == "__main__":
   train_random_forest()