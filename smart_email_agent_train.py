"""
smart_email_agent_train.py

- Trains a MultinomialNB on the provided bag-of-words Kaggle CSV (emails.csv).
- Saves:
    - email_classifier.pkl  -> trained MultinomialNB model
    - vocab.pkl             -> list of vocabulary words (column order used)
    - classes.pkl           -> list of classes the model was trained on
Notes:
- Assumes emails.csv has a header and a 'Prediction' column (labels can be 0/1 or any ints).
- It will try to drop an 'Email No.'-like id column if present.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# CONFIG
CSV_PATH = "emails.csv"
MODEL_PATH = "email_classifier.pkl"
VOCAB_PATH = "vocab.pkl"
CLASSES_PATH = "classes.pkl"
TEST_SIZE = 0.20
RANDOM_STATE = 42

def main():
    p = Path(CSV_PATH)
    if not p.exists():
        raise FileNotFoundError(f"Could not find {CSV_PATH} in current directory.")

    print("Loading dataset:", CSV_PATH)
    df = pd.read_csv(CSV_PATH)
    print("Raw shape:", df.shape)

    # Try to drop identifier columns if present
    drop_candidates = ["Email No.", "Email", "Email_No", "EmailNo", "Email No", "email no.", "email_no"]
    drop_cols = [c for c in drop_candidates if c in df.columns]
    if drop_cols:
        print("Dropping identifier columns:", drop_cols)

    # Expect a 'Prediction' column for labels
    if "Prediction" not in df.columns:
        raise ValueError("Dataset must contain a 'Prediction' column with labels (integers).")

    X = df.drop(columns=drop_cols + ["Prediction"], errors='ignore')
    y = df["Prediction"]

    # Convert y to integer labels if possible
    try:
        y = y.astype(int)
    except Exception:
        # leave as-is if non-integer labels (sklearn accepts strings too)
        pass

    print("Feature matrix shape (rows, cols):", X.shape)
    classes = np.unique(y)
    print("Classes detected:", classes)

    # Save vocabulary (column order matters)
    vocab = list(X.columns)
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump(vocab, f)
    print(f"Saved vocabulary ({len(vocab)} words) to {VOCAB_PATH}")

    # Convert features to numpy (counts)
    X_np = X.values.astype(float)

    # Train/test split (stratify if possible)
    stratify = y if len(classes) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y.values, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify
    )

    # Train MultinomialNB (use partial_fit with classes)
    clf = MultinomialNB()
    clf.partial_fit(X_train, y_train, classes=classes)
    print("Model training completed.")

    # Evaluate
    y_pred = clf.predict(X_test)
    print("\nEvaluation on test set:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    # Save model and classes
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)
    with open(CLASSES_PATH, "wb") as f:
        pickle.dump(list(classes), f)

    print(f"Saved model to '{MODEL_PATH}' and classes to '{CLASSES_PATH}'.")
    print("Training finished successfully.")

if __name__ == "__main__":
    main()
