import sys
from sentence_transformers import SentenceTransformer
import joblib
from datetime import date

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

# print(predictions)

# create variable that has output filename in format

today = date.today()
output_filename = f"headlines_scored_{headline_source}_{today.year}_{today.month}_{today.day}.txt"

# print(output_filename)

with open(output_filename, "w", encoding="utf-8") as f:
    for headline, pred in zip(headlines, predictions):
        f.write(f"{pred}, {headline}\n")