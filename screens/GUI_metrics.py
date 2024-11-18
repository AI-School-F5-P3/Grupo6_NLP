import streamlit as st
import os

def model_metrics_screen():
    st.title("Model Metrics and Performance")

    # Load confusion matrix
    current_dir = os.path.dirname(os.path.abspath(__file__))
    confusion_matrix_path = os.path.join(current_dir, '..', 'src', 'metrics', 'Naive_Bayes_SVD_confusion_matrix.png')
    
    if os.path.exists(confusion_matrix_path):
        st.image(confusion_matrix_path, caption="Confusion Matrix")
    else:
        st.warning("Confusion matrix image not found.")

    # Load and display feature importance
    feature_importance_path = os.path.join(current_dir, '..', 'src', 'metrics', 'feature_importance.png')
    if os.path.exists(feature_importance_path):
        st.image(feature_importance_path, caption="Feature Importance")
    else:
        st.warning("Feature importance image not found.")

    # Display wordcloud
    wordcloud_path = os.path.join(current_dir, '..', 'src', 'metrics', 'toxic_wordcloud.png')
    if os.path.exists(wordcloud_path):
        st.image(wordcloud_path, caption="Word Cloud of Toxic Comments")
    else:
        st.warning("Word cloud image not found.")