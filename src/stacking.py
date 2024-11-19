import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC
from sklearn.decomposition import NMF
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, balanced_accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump, load
import os
import torch
from transformers import BertTokenizer, BertModel

class BertVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, texts):
        self.model.eval()
        embeddings = []
        
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                embeddings.append(embedding[0])
        
        return np.array(embeddings)

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
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, '..', 'src', 'metrics', f'{model_name}_bert_confusion_matrix.png')
    plt.savefig(file_path)
    plt.close()

    return model, overfitting

def create_base_models():
    lr = LogisticRegression(
        C=0.05,
        class_weight='balanced',
        max_iter=2000,
        random_state=42,
        penalty='l2'
    )
    
    svm = SVC(
        kernel='rbf',
        C=0.1,
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=42
    )
    
    return [('lr', lr), ('svm', svm)]

def train_ensemble():
    df = load_data('model_df.csv')
    
    # Get BERT embeddings
    print("Generating BERT embeddings...")
    bert_vectorizer = BertVectorizer()
    X = bert_vectorizer.transform(df['processed_text'])
    y = df['IsToxic'].astype(int)
    
    # Scale the BERT embeddings to non-negative values
    print("Scaling embeddings to non-negative values...")
    minmax_scaler = MinMaxScaler()
    X_non_negative = minmax_scaler.fit_transform(X)
    
    # Apply NMF
    print("Applying NMF...")
    nmf = NMF(
        n_components=50,
        random_state=42,
        init='nndsvd',
        max_iter=300,
        tol=0.01
    )
    X = nmf.fit_transform(X_non_negative)
    
    # Split the data before scaling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train individual models
    print("\nTraining individual models...")
    
    # Train ComplementNB separately on non-negative data
    nb_model = ComplementNB(alpha=1.5, norm=True)
    nb_model.fit(X_train, y_train)
    
    # Scale data for other models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train base models for ensemble
    base_models = create_base_models()
    
    meta_clf = LogisticRegression(
        C=0.05,
        random_state=42,
        class_weight='balanced'
    )
    
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_clf,
        cv=10,
        n_jobs=-1,
        passthrough=True
    )
    
    voting = VotingClassifier(
        estimators=base_models,
        voting='soft',
        weights=[2, 3]
    )
    
    # Train ensemble models
    print("\nTraining Stacking Classifier...")
    stacking_model, stacking_overfit = train_evaluate_model(
        stacking, X_train_scaled, X_test_scaled, y_train, y_test, "Stacking_Ensemble_BERT"
    )
    
    print("\nTraining Voting Classifier...")
    voting_model, voting_overfit = train_evaluate_model(
        voting, X_train_scaled, X_test_scaled, y_train, y_test, "Voting_Ensemble_BERT"
    )
    
    # Train and evaluate NaiveBayes separately
    print("\nTraining Naive Bayes Classifier...")
    nb_overfit = train_evaluate_model(
        nb_model, X_train, X_test, y_train, y_test, "NaiveBayes_BERT"
    )[1]
    
    # Determine best model
    models = {
        'stacking': (stacking_model, stacking_overfit),
        'voting': (voting_model, voting_overfit),
        'naive_bayes': (nb_model, nb_overfit)
    }
    
    best_name = min(models.items(), key=lambda x: x[1][1])[0]
    best_model = models[best_name][0]
    
    # Save models and transformers
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'models')
    os.makedirs(model_path, exist_ok=True)
    
    print("\nSaving models and transformers...")
    try:
        dump(best_model, os.path.join(model_path, f'ensemble_{best_name}_bert_model.joblib'))
        dump(bert_vectorizer, os.path.join(model_path, 'ensemble_bert_vectorizer.joblib'))
        dump(minmax_scaler, os.path.join(model_path, 'ensemble_bert_minmax_scaler.joblib'))
        dump(nmf, os.path.join(model_path, 'ensemble_bert_nmf.joblib'))
        dump(scaler, os.path.join(model_path, 'ensemble_bert_scaler.joblib'))
        print(f"\nBest BERT-based model ({best_name}) and all transformers saved successfully.")
    except Exception as e:
        print(f"\nError saving models: {str(e)}")
        print("Training completed but models could not be saved.")
    
    return best_model

if __name__ == "__main__":
    train_ensemble()