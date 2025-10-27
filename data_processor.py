import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# --- 1. SETUP ---
# NOTE: If you haven't run nltk.download('stopwords') recently, uncomment and run it once.
# try:
#     nltk.download('stopwords')
# except:
#     pass

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Performs tokenization, stopword removal, and stemming on task summaries."""
    text = str(text).lower()
    # Remove characters that aren't letters or spaces
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    # Remove stopwords and apply stemming
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def run_data_preprocessing(input_file="D:\Data Science Intern\GFG_FINAL.csv", output_file="processed_task_data_W1.csv"):
    # --- 2. DATA LOADING & INITIAL CLEANING ---
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Selecting the key columns needed for classification (Task, Target, and a few IDs)
    key_columns = ['Summary', 'Issue Type', 'Status', 'Issue id', 'Project key'] 
    df_clean = df[key_columns].copy()
    
    # Handling missing values in the critical 'Summary' column
    initial_rows = df_clean.shape[0]
    df_clean.dropna(subset=['Summary'], inplace=True) 
    print(f"Dropped {initial_rows - df_clean.shape[0]} rows missing the task summary.")
    
    # Filling missing values in the 'Status' column
    df_clean['Status'].fillna('UNKNOWN', inplace=True) 

    # --- 3. NLP PREPROCESSING ---
    print("Applying NLP preprocessing to task summaries...")
    df_clean['Processed_Summary'] = df_clean['Summary'].apply(preprocess_text)

    # --- 4. EDA/Feature Engineering for Week 2 preparation ---
    df_clean['Summary_Length'] = df_clean['Summary'].apply(len)
    
    # --- 5. SAVING OUTPUT ---
    df_clean.to_csv(output_file, index=False)
    print(f"Successfully saved cleaned and preprocessed data (shape: {df_clean.shape}) to {output_file}")
    
    return df_clean

if __name__ == "__main__":
    run_data_preprocessing()