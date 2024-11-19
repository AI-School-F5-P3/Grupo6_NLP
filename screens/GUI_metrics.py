import streamlit as st
import os

def model_metrics_screen():
    st.title("Model Metrics and Performance")

    # Load confusion matrix
    current_dir = os.path.dirname(os.path.abspath(__file__))
    confusion_matrix_path = os.path.join(current_dir, '..', 'src', 'metrics', 'Stacking_Ensemble_BERT_bert_confusion_matrix.png')
    
    if os.path.exists(confusion_matrix_path):
        st.image(confusion_matrix_path, caption="Confusion Matrix")
    else:
        st.warning("Confusion matrix image not found.")

    # Display wordcloud
    wordcloud_path = os.path.join(current_dir, '..', 'src', 'metrics', 'toxic_wordcloud.png')
    if os.path.exists(wordcloud_path):
        st.image(wordcloud_path, caption="Word Cloud of Toxic Comments")
    else:
        st.warning("Word cloud image not found.")