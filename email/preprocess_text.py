import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

# Download stopwords
nltk.download('stopwords')

# Load stopwords from NLTK
stop_words = set(stopwords.words('english'))

# Define the preprocessing function
def preprocess_text(text):
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load the dataset without headers
data = pd.read_csv('spam.csv', encoding='latin-1', header=None, on_bad_lines='skip')

# Assign column names
data.columns = ['label', 'message']

# Fill missing values in 'message' column with empty strings
data['message'] = data['message'].fillna('')

# Encode labels (spam=1, ham=0)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Preprocess messages
data['message'] = data['message'].apply(preprocess_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)
