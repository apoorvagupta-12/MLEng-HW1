import sys
from sentence_transformers import SentenceTransformer
import joblib

# Check if we have the right number of arguments
if len(sys.argv) != 3:
    print("Usage: python score_headlines.py <file_with_headlines.txt> <source of headlines>")
    sys.exit(1)

# Get the parameters
headline_file_name = sys.argv[1]
headline_source = sys.argv[2]

# print(headline_file_name)
# print(headline_source)

# open file as txt and convert headlines to vectors

try:
    with open(headline_file_name, 'r', encoding='utf-8') as f:
        headlines = [line.strip() for line in f if line.strip()]
except FileNotFoundError:
    print(f"File {headline_file_name} not found.")
    sys.exit(1)

encoder = SentenceTransformer("all-MiniLM-L6-v2")
headline_vectors = encoder.encode(headlines)

# print(headlines)

# load model

model = joblib.load('./assignment/svm.joblib')

# make predictions

predictions = model.predict(headline_vectors)

print(predictions)

