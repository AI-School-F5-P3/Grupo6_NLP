import pandas as pd
import spacy
import re
import os
import emoji
import unicodedata

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def remove_emojis(text):
    # Remove emojis and symbols
    text = emoji.replace_emoji(text, replace='')
    
    # Remove remaining unicode symbols and pictographs
    cleaned_text = ''.join(c for c in text if not unicodedata.category(c).startswith('So'))
    
    return cleaned_text

def preprocess_text(text):
    # Remove emojis
    text = remove_emojis(text)
    
    # Process with spaCy
    doc = nlp(text)
    
    # Tokenize, lowercase, remove punctuation and stopwords
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
    
    return " ".join(tokens)

def clean_data(input_file, output_file):
    # Load the dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', input_file)
    df = pd.read_csv(data_path)

    # Apply preprocessing to the 'Text' column
    df['processed_text'] = df['Text'].apply(preprocess_text)

    # Save the preprocessed data
    output_path = os.path.join(os.path.dirname(data_path), output_file)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    clean_data('youtube.csv', 'preprocessed_data.csv')