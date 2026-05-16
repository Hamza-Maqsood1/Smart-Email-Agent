# 📧 Smart Email Classification Agent

![NLP](https://img.shields.io/badge/NLP-TF--IDF%20%2F%20Bag--of--Words-blue)
![Model](https://img.shields.io/badge/Model-Multinomial%20Naive%20Bayes-orange)
![UI](https://img.shields.io/badge/UI-Chainlit-purple)
![Framework](https://img.shields.io/badge/Framework-Scikit--learn-green)

## 📌 Overview

An NLP-powered email classification system that automatically categorizes incoming emails using classical machine learning. The system preprocesses raw email text, extracts features, and classifies emails in real time through an interactive **Chainlit** chat interface.

---

## 🎯 Problem Statement

- Manual email sorting is time-consuming and error-prone
- Goal: Build a lightweight, deployable classifier that categorizes emails accurately without heavy transformer models
- Emphasis on speed and interpretability over complexity

---

## 🔬 Methodology

### Pipeline
```
Raw Email Text → Preprocessing → Feature Extraction → Classification → Chainlit UI
```

### Text Preprocessing (NLTK)
- Stopword removal
- Tokenization
- Stemming

### Feature Extraction
- **Bag-of-Words** representation from Kaggle email dataset
- Vocabulary saved as `vocab.pkl` for inference reuse

### Model
- **Multinomial Naive Bayes** well-suited for discrete word count features
- Trained with stratified 80/20 train-test split
- Model serialized as `email_classifier.pkl` for deployment

### Deployment
- **Chainlit** interactive chat UI user pastes email text, gets instant classification

---

## 📁 Repository Structure

```
├── smart_email_agent_train.py  # Training script
├── app.py                      # Chainlit deployment app
├── email_classifier.pkl        # Trained Naive Bayes model
├── vocab.pkl                   # Vocabulary (feature columns)
├── classes.pkl                 # Class labels
└── README.md                   # Documentation
```

---

## 🚀 How to Run

### Train the Model
```bash
# Install dependencies
pip install scikit-learn nltk pandas chainlit

# Place emails.csv (Kaggle dataset) in project root
# Run training
python smart_email_agent_train.py
```

### Run the Chainlit App
```bash
chainlit run app.py
```

---

## 🛠️ Tech Stack

- **Language:** Python
- **ML:** Scikit-learn (MultinomialNB)
- **NLP:** NLTK (tokenization, stopwords, stemming)
- **UI:** Chainlit
- **Dataset:** Kaggle Email Dataset (Bag-of-Words format)

---

## 🔮 Future Work

- Upgrade to TF-IDF vectorization for better feature representation
- Add multi-class support (Work, Personal, Spam, Promotions)
- Replace Naive Bayes with fine-tuned DistilBERT for higher accuracy
- Deploy as REST API with FastAPI

---

## 👤 Author

**Hamza Maqsood**
BS Artificial Intelligence University of Management and Technology, Lahore
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/hamza-maqsood1)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/Hamza-Maqsood1)
