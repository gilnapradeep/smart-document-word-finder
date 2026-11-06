# Smart Document Word Finder

## Overview
Smart Document Word Finder is an NLP-based application that helps users discover related words in PDF and TXT documents.

Instead of relying only on exact keyword matching, it combines Word2Vec and TF-IDF-based word distribution similarity to rank related terms.

## Features
- Upload PDF and TXT documents
- Extract text from documents
- Preprocess text using NLTK
- Tokenization, stopword removal and lemmatization
- Train a Word2Vec model on document text
- Calculate TF-IDF-based word distribution similarity
- Rank related words using a combined score
- Visualize results with a chart
- Download results as a CSV file

## Technologies Used
- Python
- Streamlit
- NLTK
- Gensim
- Scikit-learn
- Pandas
- NumPy
- PyMuPDF
- Matplotlib

## Project Structure
```text
smart-document-word-finder/
├── app.py
├── preprocessing.py
├── document_reader.py
├── similarity.py
├── requirements.txt
├── README.md
├── .gitignore
└── data/
    └── sample_document.txt