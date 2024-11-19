import streamlit as st
import joblib
import os
from src.data_cleanning import preprocess_text
from src.stacking import predict_toxicity_stacking

def load_stacking_model_and_transformers():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'src', 'models', 'ensemble_stacking_bert_model.joblib')
    bert_vectorizer_path = os.path.join(current_dir, '..', 'src', 'models', 'ensemble_bert_vectorizer.joblib')
    minmax_scaler_path = os.path.join(current_dir, '..', 'src', 'models', 'ensemble_bert_minmax_scaler.joblib')
    nmf_path = os.path.join(current_dir, '..', 'src', 'models', 'ensemble_bert_nmf.joblib')
    scaler_path = os.path.join(current_dir, '..', 'src', 'models', 'ensemble_bert_scaler.joblib')

    model = joblib.load(model_path)
    bert_vectorizer = joblib.load(bert_vectorizer_path)
    minmax_scaler = joblib.load(minmax_scaler_path)
    nmf = joblib.load(nmf_path)
    scaler = joblib.load(scaler_path)

    return model, bert_vectorizer, minmax_scaler, nmf, scaler

def home_screen():
    st.title("YouTube Comment Toxicity Detector 🕵️‍♀️")
    st.markdown("""
    ### Detect Toxic Comments in Seconds!
    
    This app uses advanced Natural Language Processing to analyze the toxicity of YouTube comments.
    Simply paste a comment below and click "Analyze Toxicity" to get instant results.
    
    ---
    """)

    model, bert_vectorizer, minmax_scaler, nmf, scaler = load_stacking_model_and_transformers()

    user_input = st.text_area(
        "Enter a YouTube comment to analyze:", 
        placeholder="Paste your comment here...",
        height=200
    )

    if st.button("Analyze Toxicity", type="primary"):
        if user_input:
            processed_text, _ = preprocess_text(user_input)
            prediction, probability = predict_toxicity_stacking(processed_text, model, bert_vectorizer, minmax_scaler, nmf, scaler)
            
            if prediction == 1:
                st.error(f"🚨 Toxic Comment Detected!")
                st.warning(f"Toxicity Probability: {probability:.2%}")
                st.info("This comment may violate community guidelines. Consider reporting or moderating.")
            else:
                st.success("✅ Non-Toxic Comment")
                st.info(f"Confidence: {1-probability:.2%} Safe")
        else:
            st.warning("Please enter a comment to analyze.")

    st.markdown("---")
    with st.expander("About the Toxicity Detector"):
        st.write("""
        ### How It Works
        - Uses a Stacking Ensemble model with BERT embeddings
        - Trained on YouTube comment data
        - Analyzes text for potentially harmful content
        - Provides toxicity probability
        
        ### Limitations
        - Trained on specific dataset
        - May not catch all nuanced forms of toxicity
        - Continuous improvement needed
        """)