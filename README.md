🧠 ML-Powered Fact-Checking System

A Machine Learning + NLP-based web application that automatically classifies political statements and online claims as True or False, combining real-time data, linguistic analysis, and robust model evaluation.

🚀 Overview

This project implements a production-style NLP pipeline for automated fact-checking. It integrates data scraping, feature engineering, model training, and live benchmarking into a single interactive application built with Streamlit.

Unlike basic classifiers, this system focuses on linguistic intelligence + real-world validation, making it closer to real-world ML systems used in misinformation detection.

⚙️ Key Features
🔍 End-to-End Data Pipeline
Scrapes historical fact-check data from PolitiFact
Integrates Google Fact Check Tools API for real-time claims
Enables live benchmarking of model performance
🧾 Advanced NLP Feature Engineering

Extracts 5 levels of linguistic features:

Lexical & Morphological → vocabulary patterns
Syntactic → POS tags, sentence structure
Semantic → sentiment & subjectivity (TextBlob)
Discourse → argument structure
Pragmatic → intent-based keywords

🤖 Machine Learning Models

Trained multiple classifiers:

Logistic Regression
Support Vector Machine (SVM)
Decision Trees
Naive Bayes

📊 Robust Training Strategy
Stratified K-Fold Cross Validation
SMOTE for class imbalance handling
Performance optimization across multiple metrics

📈 Interactive Evaluation Dashboard
Accuracy, Precision, Recall, F1-score
Inference latency comparison
Real-time benchmarking with live data
Fun model critique generator explaining predictions

🧱 Tech Stack
Languages: Python
Frontend: Streamlit
ML/NLP: Scikit-learn, SpaCy, TextBlob
Data Handling: Pandas, NumPy
Imbalance Handling: Imbalanced-learn (SMOTE)
Web Scraping: BeautifulSoup
APIs: Google Fact Check Tools API
