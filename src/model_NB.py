import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from wordcloud import WordCloud
import optuna
# Import nltk and download required resources
import nltk
from nltk.corpus import wordnet
import random
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('punkt_tab')

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

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def synonym_replacement(text, n=1):
    words = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(words)
    new_words = words.copy()
    
    replacement_count = 0
    for i, (word, pos) in enumerate(pos_tags):
        if replacement_count >= n:
            break
        
        wordnet_pos = get_wordnet_pos(pos)
        if not wordnet_pos:
            continue
        
        synsets = wordnet.synsets(word, pos=wordnet_pos)
        if not synsets:
            continue
        
        synonyms = []
        for syn in synsets:
            for lemma in syn.lemmas():
                if lemma.name().lower() != word.lower():
                    synonyms.append(lemma.name())
        
        if synonyms:
            new_words[i] = random.choice(synonyms)
            replacement_count += 1
    
    return ' '.join(new_words)

def train_naive_bayes():
    # Load preprocessed data
    df = load_data('model_df.csv')

    # Ensure 'IsToxic' is binary 
    df['IsToxic'] = df['IsToxic'].astype(int)
    
    # Create a word cloud for toxic comments before vectorization
    create_wordcloud(df)
    
    # Data augmentation with WordNet-based synonym replacement
    augmented_texts = []
    augmented_labels = []
    for _, row in df.iterrows():
        augmented_texts.append(row['processed_text'])
        augmented_labels.append(row['IsToxic'])
        if row['IsToxic'] == 1:  # Augment only toxic comments
            augmented_text = synonym_replacement(row['processed_text'], n=2)
            augmented_texts.append(augmented_text)
            augmented_labels.append(1)
    
    # Create a new DataFrame with original and augmented data
    df_augmented = pd.DataFrame({'processed_text': augmented_texts, 'IsToxic': augmented_labels})

    # Confirm 'IsToxic' is integer type
    df_augmented['IsToxic'] = df_augmented['IsToxic'].astype(int)

    # Vectorization using TF-IDF with reduced max_features, min_df, and bigrams
    vectorizer = TfidfVectorizer(max_features=2000, min_df=5, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df_augmented['processed_text'])
    y = df_augmented['IsToxic'].astype(int)

    # Split the data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Apply SMOTE on training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Calculate class weights for imbalance
    class_weight_dict = compute_class_weight('balanced', classes=np.unique(y_train_smote), y=y_train_smote)
    class_prior = [class_weight_dict[0] / sum(class_weight_dict), class_weight_dict[1] / sum(class_weight_dict)]

    # Feature Selection with Chi-Square Test
    selector = SelectKBest(chi2, k=1000)
    X_train_selected = selector.fit_transform(X_train_smote, y_train_smote)
    X_test_selected = selector.transform(X_test)

    # Define the Optuna optimization function
    def objective(trial):
        alpha = trial.suggest_loguniform('alpha', 0.01, 5.0)
        fit_prior = trial.suggest_categorical('fit_prior', [True, False])
        model = ComplementNB(alpha=alpha, fit_prior=fit_prior)
        
        # Stratified K-Fold Cross-Validation
        skf = StratifiedKFold(n_splits=5)
        scores = cross_val_score(model, X_train_selected, y_train_smote, cv=skf, scoring='roc_auc')
        return scores.mean()
    
    # Run Optuna optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    best_alpha = study.best_params['alpha']
    best_fit_prior = study.best_params['fit_prior']
    print(f"Best alpha found by Optuna: {best_alpha}")
    print(f"Best fit_prior found by Optuna: {best_fit_prior}")
    
    # Train final model with best alpha from Optuna
    best_model = ComplementNB(alpha=best_alpha, fit_prior=best_fit_prior)
    best_model.fit(X_train_selected, y_train_smote)
    
    # Evaluate the model
    best_model = train_evaluate_model(best_model, X_train_selected, X_test_selected, y_train_smote, y_test, "Complement_Naive_Bayes")
    
    # Feature Importance Analysis (for Naive Bayes, we'll use the log probabilities)
    feature_importance = best_model.feature_log_prob_[1] - best_model.feature_log_prob_[0]
    selected_feature_names = selector.get_feature_names_out(vectorizer.get_feature_names_out())
    importance_df = pd.DataFrame({'feature': selected_feature_names, 'importance': feature_importance})
    importance_df = importance_df.sort_values('importance', ascending=False).head(20)

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
