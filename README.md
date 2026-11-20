# SMS Spam Detector 📩

A supervised Machine Learning text classification project built in Python using **Scikit-Learn** and the **Multinomial Naive Bayes** algorithm to filter spam messages from legitimate texts.

## How It Works
1. **Text Vectorization:** Uses `CountVectorizer` (Bag-of-Words) to convert raw text messages into numerical frequency matrices.
2. **Model Training:** Trains a `MultinomialNB` classifier on labeled spam and non-spam text samples.
3. **Inference:** Predicts whether incoming messages are `SPAM` or `NOT SPAM`.

## Tech Stack
- **Language:** Python 3
- **Library:** Scikit-Learn (`sklearn`)

## How to Run
```bash
pip install scikit-learn
python3 spam_detector.py
Message: 'Claim your free lottery reward now!' --> Prediction: SPAM
Message: 'Can you share the GitHub link for our assignment?' --> Prediction: NOT SPAM
