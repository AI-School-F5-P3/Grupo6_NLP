import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from src.data_cleanning import preprocess_text
from src.stacking import predict_toxicity_stacking, BertVectorizer
import joblib
import os
from dotenv import load_dotenv
import cloudpickle

def load_stacking_model_and_transformers():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Change file extensions to .pkl
    model_path = os.path.join(current_dir, '..', 'src', 'models', f'ensemble_stacking_bert_model.pkl')
    bert_vectorizer_path = os.path.join(current_dir, '..', 'src', 'models', 'ensemble_bert_vectorizer.pkl')
    minmax_scaler_path = os.path.join(current_dir, '..', 'src', 'models', 'ensemble_bert_minmax_scaler.pkl')
    nmf_path = os.path.join(current_dir, '..', 'src', 'models', 'ensemble_bert_nmf.pkl')
    scaler_path = os.path.join(current_dir, '..', 'src', 'models', 'ensemble_bert_scaler.pkl')

    # Use cloudpickle.load instead of joblib.load
    with open(model_path, 'rb') as f:
        model = cloudpickle.load(f)
    
    with open(bert_vectorizer_path, 'rb') as f:
        bert_vectorizer = cloudpickle.load(f)
    
    with open(minmax_scaler_path, 'rb') as f:
        minmax_scaler = cloudpickle.load(f)
    
    with open(nmf_path, 'rb') as f:
        nmf = cloudpickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = cloudpickle.load(f)

    return model, bert_vectorizer, minmax_scaler, nmf, scaler

# Load environment variables
load_dotenv()

# YouTube API setup
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_video_comments(video_id, max_comments=50):
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=min(100, max_comments)  
        )
        while request and len(comments) < max_comments:
            response = request.execute()
            for item in response["items"]:
                if len(comments) >= max_comments:
                    break
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)
            request = youtube.commentThreads().list_next(request, response)
    except HttpError as e:
        st.error(f"An error occurred: {e}")
    return comments[:max_comments] 

def analyze_comments(comments, model, bert_vectorizer, minmax_scaler, nmf, scaler):
    results = []
    for comment in comments:
        processed_text, _ = preprocess_text(comment)
        prediction, probability = predict_toxicity_stacking(processed_text, model, bert_vectorizer, minmax_scaler, nmf, scaler)
        results.append({
            "comment": comment,
            "processed_text": processed_text,
            "is_toxic": prediction,
            "toxicity_probability": probability
        })
    return results

def youtube_analyzer_screen():
    st.title("YouTube Comment Toxicity Analyzer")
    
    video_url = st.text_input("Enter YouTube Video URL:")
    
    if st.button("Analyze Comments"):
        if video_url:
            try:
                video_id = video_url.split("v=")[1]
                comments = get_video_comments(video_id)
                
                if comments:
                    model, bert_vectorizer, minmax_scaler, nmf, scaler = load_stacking_model_and_transformers()
                    results = analyze_comments(comments, model, bert_vectorizer, minmax_scaler, nmf, scaler)
                    df = pd.DataFrame(results)
                    
                    st.subheader("Analysis Results")
                    st.write(f"Total comments analyzed: {len(df)}")
                    st.write(f"Toxic comments found: {df['is_toxic'].sum()}")
                    
                    st.subheader("Toxic Comments")
                    toxic_comments = df[df['is_toxic'] == 1].sort_values('toxicity_probability', ascending=False)
                    st.dataframe(toxic_comments[['comment', 'toxicity_probability']])
                    
                    st.subheader("Non-Toxic Comments")
                    non_toxic_comments = df[df['is_toxic'] == 0].sort_values('toxicity_probability')
                    st.dataframe(non_toxic_comments[['comment', 'toxicity_probability']])
                else:
                    st.write("No comments found or unable to fetch comments.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.write("Please enter a valid YouTube video URL.")