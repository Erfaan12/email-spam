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

# Step 1: Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Step 2: Define text preprocessing function
def preprocess_text(text):
    # Remove punctuation
    text = ''.join([char for char in text if char not in string.punctuation])
    # Convert to lowercase
    text = text.lower()
    # Remove stopwords
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Step 3: Load the dataset as a tab-separated file
data = pd.read_csv('spam.csv', encoding='latin-1', sep='\t', header=None)

# Assign column names manually since there's no header
data.columns = ['label', 'message']

# Step 4: Convert the 'label' column to numeric values (spam=1, ham=0)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Step 5: Preprocess the messages
data['message'] = data['message'].apply(preprocess_text)

# Step 6: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Step 7: Build and train the model using a pipeline
pipeline = Pipeline([
    ('vect', CountVectorizer()),           # Convert text to token counts
    ('tfidf', TfidfTransformer()),         # Convert counts to TF-IDF features
    ('clf', MultinomialNB())               # Train a Naive Bayes classifier
])
pipeline.fit(X_train, y_train)

# Step 8: Evaluate the model
y_pred = pipeline.predict(X_test)
print(f'Accuracy: {metrics.accuracy_score(y_test, y_pred)}')
print(metrics.classification_report(y_test, y_pred, target_names=['ham', 'spam']))

# Step 9: Save the trained model
joblib.dump(pipeline, 'spam_detector.pkl')

# Step 10: Load the model (for future use)
pipeline = joblib.load('spam_detector.pkl')

# Step 11: Test the model with new data
new_emails = [
    "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.", 
    "Hey Bob, can we schedule a meeting for tomorrow?"
]
# Preprocess the new emails
new_emails = [preprocess_text(email) for email in new_emails]
# Predict whether the new emails are spam or ham
predictions = pipeline.predict(new_emails)
print(predictions)  # Output: array([1, 0]) indicating [spam, ham]
