import streamlit as st
from screens.GUI_home import home_screen
from screens.GUI_metrics import model_metrics_screen
from screens.GUI_youtube_analyzer import youtube_analyzer_screen

def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Model Metrics", "YouTube Analyzer"])

    if page == "Home":
        home_screen()
    elif page == "Model Metrics":
        model_metrics_screen()
    elif page == "YouTube Analyzer":
        youtube_analyzer_screen()

if __name__ == "__main__":
    st.set_page_config(page_title="Toxicity Detector", page_icon=":warning:", layout="centered")
    main()