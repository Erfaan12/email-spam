import pandas as pd
from sklearn.model_selection import train_test_split

# Load the dataset with the correct delimiter
data = pd.read_csv('spam.csv', encoding='latin-1', delimiter='\t', on_bad_lines='skip')

# Inspect the data
print(data.head())
print(data.columns)

# Select and rename columns based on the actual CSV structure
if data.shape[1] == 2:  # Check if there are exactly two columns
    data.columns = ['label', 'message']
else:
    print("Unexpected number of columns. Please check the CSV file structure.")
    exit()

# Encode labels (spam=1, ham=0)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(data['message'], data['label'], test_size=0.2, random_state=42)

# Print the shapes to verify
print(f'Training set: {X_train.shape}, {y_train.shape}')
print(f'Test set: {X_test.shape}, {y_test.shape}')
