import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

INPUT_FILE = "processed_task_data_W1.csv"
TARGET_COLUMN = 'Issue Type'
TEXT_COLUMN = 'Processed_Summary'

def feature_extraction_and_modeling():
    """Performs Feature Extraction (TF-IDF) and trains Classification Models."""
    print("--- Starting Week 2: Feature Extraction and Modeling ---")
    
    # --- 1. Load Processed Data ---
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Input file '{INPUT_FILE}' not found. Ensure data_processor.py was run.")
        return

    # Filter out any classes that are too small for reliable training
    min_samples = 10 
    valid_classes = df[TARGET_COLUMN].value_counts() > min_samples
    valid_classes = valid_classes[valid_classes].index.tolist()
    df = df[df[TARGET_COLUMN].isin(valid_classes)]
    
    X = df[TEXT_COLUMN].astype(str)
    y = df[TARGET_COLUMN]

    # --- 2. Feature Extraction (TF-IDF) ---
    print("Applying TF-IDF Feature Extraction...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)
    
    # Split Data (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X_tfidf, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- 3. Model Training and Evaluation (Classification) ---
    
    # 3a. Naive Bayes Classifier
    print("\nTraining Naive Bayes (MultinomialNB)...")
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    
    print("\n--- Naive Bayes Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, nb_pred):.4f}")
    # Note: Use weighted average for precision/recall in multi-class classification
    print(f"Weighted Precision: {precision_score(y_test, nb_pred, average='weighted', zero_division=0):.4f}")
    print(f"Weighted Recall: {recall_score(y_test, nb_pred, average='weighted', zero_division=0):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, nb_pred, zero_division=0))

    # 3b. Support Vector Machine (LinearSVC)
    print("\nTraining Support Vector Machine (LinearSVC)...")
    svm_model = LinearSVC(random_state=42, dual=False)
    svm_model.fit(X_train, y_train)
    svm_pred = svm_model.predict(X_test)
    
    print("\n--- SVM Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, svm_pred):.4f}")
    print(f"Weighted Precision: {precision_score(y_test, svm_pred, average='weighted', zero_division=0):.4f}")
    print(f"Weighted Recall: {recall_score(y_test, svm_pred, average='weighted', zero_division=0):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, svm_pred, zero_division=0))

if __name__ == "__main__":
    feature_extraction_and_modeling()