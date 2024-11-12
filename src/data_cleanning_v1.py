import pandas as pd
import spacy
import re
import os
import emoji
import unicodedata

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def remove_emojis(text):
    # Removes emojis
    text = emoji.replace_emoji(text, replace='')  # Removes emojis directly
    cleaned_text = ''.join(c for c in text if not unicodedata.category(c).startswith('So'))  # Remove unicode pictographs
    return cleaned_text

def preprocess_text(text):
    # Step 1: Remove emojis
    text_no_emoji = remove_emojis(text)
    
    # Step 2: Convert text to lowercase
    text_no_emoji = text_no_emoji.lower()
    
    # Step 3: Remove URLs and mentions
    text_no_emoji = re.sub(r"http\S+|www\S+|https\S+|@\S+", '', text_no_emoji)
    
    # Step 4: Process text with spaCy
    doc = nlp(text_no_emoji)
    
    # Step 5: Tokenize, remove stopwords, punctuation, and lemmatize
    tokens = [
        token.lemma_ for token in doc 
        if not token.is_stop and not token.is_punct and not token.like_num
    ]
    
    # Step 6: Join tokens into a single string
    cleaned_text = " ".join(tokens)
    
    # Step 7: Handle cases where the entire text was removed
    if cleaned_text.strip() == "":
        # Return original text without emojis if cleaned text is empty
        return text_no_emoji.strip()
    
    # Step 8: Remove extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text

def clean_data(input_file, output_file, column='Text'):
    # Load the dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, '..', 'data', input_file)
    df = pd.read_csv(data_path)

    # Check if the specified column exists
    if column not in df.columns:
        print(f"Column '{column}' not found in the dataset.")
        return

    # Apply preprocessing to the specified column
    print(f"Cleaning data in column: {column}")
    df['processed_text'] = df[column].apply(preprocess_text)

    # Save the preprocessed data
    output_path = os.path.join(os.path.dirname(data_path), output_file)
    df.to_csv(output_path, index=False)
    print(f"Preprocessed data saved to {output_path}")

if __name__ == "__main__":
    clean_data('youtube.csv', 'preprocessed_data.csv')
