# Information-Retrieval-Models
Different information retrieval models have been completed in this repository

## 📚 Overview

This repository focuses on the implementation and comparison of various **Information Retrieval (IR)** models used to retrieve relevant documents based on user queries. Information retrieval plays a critical role in search engines, question answering systems, and document classification.

The objective of this project is to analyze different strategies for representing documents and queries, and how these representations impact retrieval performance.

## 🚀 Implemented Models

Five different models were implemented:

1. **Embeddings-Based Retrieval**  
   Uses dense vector representations (e.g., Word2Vec or Sentence Transformers) to compute semantic similarity between queries and documents.

2. **Bag of Words (BoW) with TF-IDF weighting**  
   Represents documents as sparse vectors with weights based on **Term Frequency–Inverse Document Frequency**, capturing word importance across the corpus.

3. **Bag of Words (BoW) with Binary weighting**  
   A simple model where a word is marked as 1 if it appears in a document, and 0 otherwise. No weighting is applied.

4. **Bag of Words (BoW) with Rocchio Algorithm**  
   Enhances retrieval using **relevance feedback**, modifying the query vector based on relevant/non-relevant documents to improve accuracy.

## ⚙️ Technologies

- Python 3.x
- `scikit-learn` – for vectorization and similarity computation
- `numpy`, `pandas` – for data manipulation
- `nltk` or `spaCy` – for text preprocessing
- Optional: `gensim`, `sentence-transformers` – for embeddings

## 📊 Evaluation

The implemented models can be evaluated using standard Information Retrieval metrics:

- **Precision** – Measures the proportion of retrieved documents that are relevant.
- **Recall** – Measures the proportion of relevant documents that are retrieved.
- **F1-score** – Harmonic mean of precision and recall.
- **Average Precision (MAP)** – Averages the precision scores after each relevant document is retrieved.

These metrics help compare model effectiveness in real-world query scenarios.

## 🧠 Notes

- All **Bag of Words** models rely heavily on text preprocessing, including:
  - Lowercasing
  - Tokenization
  - Stopword removal
  - Stemming or lemmatization (optional, but recommended)

- The **Rocchio model** depends on a **relevance feedback mechanism**. For experimental purposes, relevance labels can be simulated or manually defined.

- **Embeddings-based retrieval** captures **semantic similarity**, making it especially effective for queries that are phrased differently than the target document's vocabulary.

- Proper corpus normalization (e.g., Unicode handling, punctuation cleanup) can significantly impact the results across all models.
