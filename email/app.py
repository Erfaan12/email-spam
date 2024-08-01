from flask import Flask, request, jsonify
import joblib
import nltk
from nltk.corpus import stopwords
import string
# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('spam_detector.pkl')

# Define text preprocessing function
def preprocess_text(text):
    text = ''.join([char for char in text if char not in string.punctuation])
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

# Define the home route
@app.route('/')
def home():
    return "<h1>Welcome to the Advanced Spam Detection API!</h1><p>Use the /predict endpoint to check for spam.</p>"

# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    # Get the email text from the request
    email = request.json['email']

    # Preprocess the email
    processed_email = preprocess_text(email)

    # Predict using the loaded model
    prediction = model.predict([processed_email])[0]
    
    # Additional features
    word_count = len(email.split())
    char_count = len(email)
    num_stopwords = len([word for word in email.split() if word.lower() in stop_words])
    num_punctuation = len([char for char in email if char in string.punctuation])
    num_uppercase_words = len([word for word in email.split() if word.isupper()])
    num_numeric_chars = sum(c.isdigit() for c in email)
    avg_word_length = sum(len(word) for word in email.split()) / len(email.split())
    keywords = ['win', 'free', 'money', 'urgent', 'prize']
    keywords_found = [word for word in keywords if word in email.lower()]
    sentence_count = email.count('.') + email.count('!') + email.count('?')
    reading_time_seconds = word_count / 200 * 60  # Assuming average reading speed of 200 words/min
    num_urls = email.count('http')
    has_greeting = any(greeting in email.lower() for greeting in ['hello', 'hi', 'dear'])
    has_signature = any(signature in email.lower() for signature in ['best regards', 'sincerely', 'cheers'])
    spam_probability = model.predict_proba([processed_email])[0][1]
    avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
    contains_digits = any(char.isdigit() for char in email)
    contains_special_chars = any(char in string.punctuation for char in email)
    contains_excessive_spaces = '  ' in email

    # Return the prediction and additional features as a JSON response
    return jsonify({
        'prediction': int(prediction),
        'spam_probability': spam_probability,
        'word_count': word_count,
        'char_count': char_count,
        'num_stopwords': num_stopwords,
        'num_punctuation': num_punctuation,
        'num_uppercase_words': num_uppercase_words,
        'num_numeric_chars': num_numeric_chars,
        'avg_word_length': avg_word_length,
        'keywords_found': keywords_found,
        'sentence_count': sentence_count,
        'reading_time_seconds': reading_time_seconds,
        'num_urls': num_urls,
        'has_greeting': has_greeting,
        'has_signature': has_signature,
        'avg_sentence_length': avg_sentence_length,
        'contains_digits': contains_digits,
        'contains_special_chars': contains_special_chars,
        'contains_excessive_spaces': contains_excessive_spaces
    })

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
