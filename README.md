# MLEng-HW1

# Headline Sentiment Scorer

This Python script classifies news headlines into the following sentiment categories: **Optimistic**, **Pessimistic**, or **Neutral** using a pretrained **SVM** model and **SentenceTransformer** embeddings.

---

## File Structure

- score_headlines.py contains the script that takes the headlines and source as input and produces a timestamped output.
- Model/svm.joblib contains the pretrained model.

---

## Requirements

Install required Python packages with:

```bash
pip install -r requirements.txt
```

## Usage

Run the script as follows

```bash
python score_headlines.py headlines.txt nyt
```

If the file with headlines and source aren't provided, an error will be thrown.
Please ensure the input txt file contains one headline per line (headlines are separated by '\n')

---

## Output

From the example usage above, the script will create a file titled headlines_nyt_yyyy_mm_dd.txt and will have the following format:

sentiment, headline

sentiment, headline



