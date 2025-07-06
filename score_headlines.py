"""Script to score headlines using a pretrained SVM given some inputs."""

import sys
from datetime import date
import joblib
from sentence_transformers import SentenceTransformer

MODEL_PATH = './assignment/svm.joblib'

def check_params():
    """ Take input from the user and ensure it's in the correct format. """

    if len(sys.argv) != 3:
        print("Usage: python score_headlines.py <file_with_headlines.txt> <source of headlines>")
        sys.exit(1)

    return sys.argv[1], sys.argv[2]

def load_headlines(filename):
    """ Load headlines from the filename provided. """

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            headlines = [line.strip() for line in f if line.strip()]
            return headlines
    except FileNotFoundError:
        print(f"File {filename} not found.")
        sys.exit(1)

def create_vectors(headlines):
    """ Create encoded vectors per guidance from loaded headlines. """

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return encoder.encode(headlines)

def load_model(model_path):
    """ Load the saved model provided. """

    return joblib.load(model_path)

def get_output_filename(source):
    """ Create output file name per standard provided. """

    today = date.today()
    return f"headlines_scored_{source}_{today.year}_{today.month}_{today.day}.txt"

def make_predictions(model, vectors):
    """ Make sentiment predictions from model. """

    return model.predict(vectors)

def save_file(filename, headlines, predictions):
    """ Save output file in the format provided. """

    with open(filename, "w", encoding="utf-8") as f:
        for headline, pred in zip(headlines, predictions):
            f.write(f"{pred}, {headline}\n")

    print("Output file is created and saved!")

def main():
    """ Main method, to call helper functions written. """

    headline_file_name, headline_source = check_params()
    headlines = load_headlines(headline_file_name)
    headline_vectors = create_vectors(headlines)
    model = load_model(MODEL_PATH)
    predictions = make_predictions(model, headline_vectors)
    output_filename = get_output_filename(headline_source)
    save_file(output_filename, headlines, predictions)


if __name__ == "__main__":
    main()
