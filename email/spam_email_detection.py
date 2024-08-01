import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics
import nltk
from nltk.corpus import stopwords
import string
import joblib

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Preprocessing function
def preprocess_text(text):
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Load the dataset
try:
    data = pd.read_csv('spam.csv', encoding='latin-1', sep='\t', names=['label', 'message'])
except pd.errors.ParserError as e:
    print(f"Error reading the CSV file: {e}")
    exit()

print("Data loaded successfully")
print(data.head())

# Drop any rows with missing values
data.dropna(inplace=True)
print("After dropping missing values")
print(data.head())

# Encode labels (spam=1, ham=0)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Drop any rows with NaN labels after encoding
data = data.dropna(subset=['label'])

print("After encoding labels and dropping NaNs")
print(data.head())

# Preprocess messages
data['message'] = data['message'].apply(preprocess_text)

print("After preprocessing messages")
print(data.head())

# Check if the DataFrame is empty
if data.empty:
    print("DataFrame is empty after preprocessing. Exiting...")
    exit()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Build and train the pipeline
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])
pipeline.fit(X_train, y_train)

# Predict on test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred)}')
print(metrics.classification_report(y_test, y_pred, target_names=['ham', 'spam']))

# Save the model
joblib.dump(pipeline, 'spam_detector.pkl')

# Test with new data
new_emails = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.",
              "Hey Bob, can we schedule a meeting for tomorrow?"]
new_emails = [preprocess_text(email) for email in new_emails]
predictions = pipeline.predict(new_emails)
print(predictions)
