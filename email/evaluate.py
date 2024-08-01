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

# Function to read and clean the CSV file
def read_and_clean_csv(filepath):
    rows = []
    with open(filepath, 'r', encoding='latin-1') as file:
        for line in file:
            parts = line.split(',', 1)  # Split on the first comma only
            if len(parts) == 2:
                label = parts[0].strip()  # Strip whitespace
                message = parts[1].strip()  # Strip whitespace
                rows.append([label, message])
            else:
                print(f"Skipping malformed line: {line}")
    return pd.DataFrame(rows, columns=['v1', 'v2'])

# Load and clean the dataset
try:
    data = read_and_clean_csv('spam.csv')
    print("Initial data shape:", data.shape)
    data = data[['v1', 'v2']]
    data.columns = ['label', 'message']
    print("Data shape after selecting columns:", data.shape)
    print("First few rows of the dataset:")
    print(data.head())
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# Print unique values in the label column before any processing
print("Unique labels before mapping:", data['label'].unique())

# Drop rows with missing values
data.dropna(inplace=True)
print("Data shape after dropping NAs:", data.shape)

# Remove special characters and whitespace from labels
data['label'] = data['label'].str.strip().str.replace(r'[^\w\s]', '', regex=True).str.lower()
print("Unique labels after stripping special characters:", data['label'].unique())

# Correctly map labels to numerical values (spam=1, ham=0)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})
print("Unique labels after mapping:", data['label'].unique())

# Check for rows where the label could not be encoded and drop them
data = data.dropna(subset=['label'])
print("Data shape after encoding labels and dropping rows with NaNs in labels:", data.shape)

# Preprocess messages
data['message'] = data['message'].apply(preprocess_text)
print("Data shape after preprocessing messages:", data.shape)

# Ensure there is data left to split
if data.shape[0] == 0:
    print("No data available after cleaning. Please check the CSV file for formatting issues.")
    exit()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)
print(f"Train set size: {len(X_train)}, Test set size: {len(X_test)}")

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
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Ensure there are at least two classes in the test set
if len(set(y_test)) > 1:
    report = metrics.classification_report(y_test, y_pred, target_names=['ham', 'spam'])
    print(report)
else:
    print("Not enough classes to generate a classification report.")
