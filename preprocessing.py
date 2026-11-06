import re
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)


stop_words = set(stopwords.words('english'))

stop_words = stop_words - {'no', 'not', 'nor'}

lemmatizer = WordNetLemmatizer()


def clean_text(text):

    
    text = text.lower()

    
    text = re.sub(r'<.*?>', ' ', text)

    
    text = re.sub(r'http\S+|www\S+', ' ', text)

    
    text = re.sub(r'[^a-z\s]', ' ', text)

    
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def preprocess_text(text):

    cleaned_text = clean_text(text)

    tokens = word_tokenize(cleaned_text)

    tokens = [
        word for word in tokens
        if word not in stop_words and len(word) > 1
    ]

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
    ]

    return tokens


def preprocess_document(text):

    sentences = sent_tokenize(text)

    processed_sentences = []

    for sentence in sentences:

        tokens = preprocess_text(sentence)

        if len(tokens) >= 2:
            processed_sentences.append(tokens)

    return processed_sentences