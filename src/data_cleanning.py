import pandas as pd
import spacy
import re
import os
import emoji
import unicodedata
from emoji import EMOJI_DATA

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def remove_emojis(text):
    text = emoji.replace_emoji(text, replace='')
    cleaned_text = ''.join(c for c in text if not unicodedata.category(c).startswith('So'))
    return cleaned_text

def transcribe_emojis(text):
    """
    Convert emojis in the text to their descriptive names (e.g., 👍 to "thumbs up").
    Only used when the text consists solely of emojis.
    """
    if not any(c.isalpha() for c in text):  # Check if text contains only emojis
        emoji_names = []
        for char in text:
            if char in EMOJI_DATA:
                emoji_name = EMOJI_DATA[char]['en'].replace('_', ' ').lower()
                emoji_names.append(emoji_name)
        return ' '.join(emoji_names)
    return text

def analyze_emoji_toxicity(emoji_text):
    """
    Analyze the toxicity of transcribed emojis.
    This is a simple implementation and can be expanded with more sophisticated analysis.
    """
    toxic_words = ['angry', 'rage', 'furious', 'mad', 'poop', 'devil', 'evil', 'skull']
    return any(word in emoji_text for word in toxic_words)

def preprocess_text(text):
    # Step 1: Transcribe emojis if text consists only of emojis
    transcribed_text = transcribe_emojis(text)
    
    if transcribed_text != text:
        # Text was transcribed, analyze its toxicity
        is_toxic = analyze_emoji_toxicity(transcribed_text)
        return transcribed_text, is_toxic

    # If text wasn't transcribed, continue with normal preprocessing
    text_no_emoji = remove_emojis(text)
    text_no_emoji = text_no_emoji.lower()
    text_no_emoji = re.sub(r"http\S+|www\S+|https\S+|@\S+", '', text_no_emoji)
    
    doc = nlp(text_no_emoji)
    
    tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct and not token.like_num
    ]
    
    cleaned_text = " ".join(tokens)
    
    if cleaned_text.strip() == "":
        return text_no_emoji.strip(), False
    
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text, False

def clean_data(input_file, output_file, column='Text'):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', input_file)
    df = pd.read_csv(data_path)

    if column not in df.columns:
        print(f"Column '{column}' not found in the dataset.")
        return

    print(f"Cleaning data in column: {column}")
    df['processed_text'], df['emoji_toxic'] = zip(*df[column].apply(preprocess_text))

    # Create a new dataframe with only the necessary columns
    focused_df = df[['processed_text', 'IsToxic']].copy()

    # Save the focused dataframe
    output_path = os.path.join(os.path.dirname(data_path), output_file)
    focused_df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

    # Save the full dataframe with all columns for potential future use
    full_output_path = os.path.join(os.path.dirname(data_path), 'preprocessed_data.csv')
    df.to_csv(full_output_path, index=False)
    print(f"Full preprocessed data saved to {full_output_path}")

def process_new_comment(text):
    """
    Process new comments from YouTube API.
    """
    processed_text, is_emoji_toxic = preprocess_text(text)
    return {
        "processed_text": processed_text,
        "is_emoji_toxic": is_emoji_toxic
    }

if __name__ == "__main__":
    clean_data('youtube.csv', 'model_df.csv')
    print("Data preprocessing completed.")