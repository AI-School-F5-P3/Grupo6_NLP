import streamlit as st
import joblib
import os
from src.data_cleanning import preprocess_text
from src.model_NB import predict_toxicity

def load_model_and_vectorizer():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'src','models', 'naive_bayes_model.joblib')
    vectorizer_path = os.path.join(current_dir, '..','src', 'models', 'tfidf_vectorizer.joblib')

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    return model, vectorizer

def home_screen():
    st.title("YouTube Comment Toxicity Detector 🕵️‍♀️")
    st.markdown("""
    ### Detect Toxic Comments in Seconds!
    
    This app uses advanced Natural Language Processing to analyze the toxicity of YouTube comments.
    Simply paste a comment below and click "Analyze Toxicity" to get instant results.
    
    ---
    """)

    model, vectorizer = load_model_and_vectorizer()

    user_input = st.text_area(
        "Enter a YouTube comment to analyze:", 
        placeholder="Paste your comment here...",
        height=200
    )

    if st.button("Analyze Toxicity", type="primary"):
        if user_input:
            processed_text, _ = preprocess_text(user_input)
            prediction, probability = predict_toxicity(processed_text, model, vectorizer)
            
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
        - Uses a Complement Naive Bayes classifier
        - Trained on YouTube comment data
        - Analyzes text for potentially harmful content
        - Provides toxicity probability
        
        ### Limitations
        - Trained on specific dataset
        - May not catch all nuanced forms of toxicity
        - Continuous improvement needed
        """)