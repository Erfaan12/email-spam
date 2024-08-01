import pandas as pd

# Load the dataset with error handling
try:
    # Specify the delimiter as tab
    data = pd.read_csv('spam.csv', encoding='latin-1', error_bad_lines=False, warn_bad_lines=True)
    print(data.head())
except Exception as e:
    print(f"Error reading the CSV file: {e}")
