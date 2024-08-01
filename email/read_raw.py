# read_raw.py
file_path = 'spam.csv'

# Read and print the first few lines of the file
with open(file_path, 'r', encoding='latin-1') as file:
    for _ in range(20):
        print(file.readline())

