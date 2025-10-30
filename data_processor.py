import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

CUSTOM_LEAKAGE_WORDS = [
    'bug', 'suggest', 'suggestion', 'bugs', 'suggestions', 
    'defect', 'new feature', 'feature', 'enhancement', 
    'task', 'story', 'epic', 'improve', 
    'srctreewin',
    'win', 'treewin' 
] 

stemmer = PorterStemmer()
STEMMED_LEAKAGE_WORDS = {stemmer.stem(word) for word in CUSTOM_LEAKAGE_WORDS}

stop_words = set(stopwords.words('english'))
stop_words.update(STEMMED_LEAKAGE_WORDS) 

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words] 
    return " ".join(tokens)

def run_data_preprocessing(input_file="GFG_FINAL.csv", output_file="processed_task_data_W1.csv"):
    # --- 2. DATA LOADING & INITIAL CLEANING ---
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)
    
    key_columns = ['Summary', 'Issue Type'] 
    df_clean = df[key_columns].copy()
    
    initial_rows = df_clean.shape[0]
    df_clean.dropna(subset=['Summary'], inplace=True) 
    print(f"Dropped {initial_rows - df_clean.shape[0]} rows missing the task summary.")

    df_clean['Summary'] = df_clean['Summary'].astype(str) 
    df_clean['Summary'] = df_clean['Summary'].apply(
        lambda x: re.sub(r'^\s*[\w\d\-_]+[\:\s]\s*', '', x)
    )

    print("Applying NLP preprocessing to task summaries...")
    df_clean['Processed_Summary'] = df_clean['Summary'].apply(preprocess_text)

    df_clean['Summary_Length'] = df_clean['Summary'].apply(len)
    
    df_clean.to_csv(output_file, index=False)
    print(f"Successfully saved cleaned and preprocessed data (shape: {df_clean.shape}) to {output_file}")
    
    return df_clean

if __name__ == "__main__":
    run_data_preprocessing()