import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from src.data_cleanning import preprocess_text
from src.model_NB import predict_toxicity
import joblib
import os
import os
from dotenv import load_dotenv

# Load the trained model and vectorizer
def load_model_and_vectorizer():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, '..', 'src', 'models', 'naive_bayes_model.joblib')
    vectorizer_path = os.path.join(current_dir, '..', 'src', 'models', 'tfidf_vectorizer.joblib')
    svd_path = os.path.join(current_dir, '..', 'src', 'models', 'svd_model.joblib')

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    svd = joblib.load(svd_path)

    return model, vectorizer, svd

# Load environment variables
load_dotenv()

# YouTube API setup
API_KEY = os.getenv("YOUTUBE_API_KEY")
youtube = build('youtube', 'v3', developerKey=API_KEY)

def get_video_comments(video_id):
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100
        )
        while request:
            response = request.execute()
            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment)
            request = youtube.commentThreads().list_next(request, response)
    except HttpError as e:
        st.error(f"An error occurred: {e}")
    return comments

def analyze_comments(comments, model, vectorizer, svd):
    results = []
    for comment in comments:
        processed_text, _ = preprocess_text(comment)
        prediction, probability = predict_toxicity(processed_text, model, vectorizer, svd)
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
                    model, vectorizer, svd = load_model_and_vectorizer()
                    results = analyze_comments(comments, model, vectorizer, svd)
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