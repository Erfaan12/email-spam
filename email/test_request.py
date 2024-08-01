import requests

# The URL where your Flask app is running
url = 'http://127.0.0.1:5000/predict'  # Use your Heroku URL after deployment

# The data you want to send to your Flask app (email text)
email_data = {
    'email': 'Hello! You have won free money! Visit http://example.com to claim your prize. Best regards.'
}

# Send the POST request to the Flask app with the email data as JSON
response = requests.post(url, json=email_data)

# Print the JSON response from the Flask app
print(response.json())
