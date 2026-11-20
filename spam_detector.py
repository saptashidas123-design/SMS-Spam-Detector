from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 1. Sample training data (messages and their labels: 1 = Spam, 0 = Normal)
messages = [
    "Congratulations! You won a free iPhone. Click here to claim now!",
    "URGENT: Your bank account is suspended. Verify your OTP immediately.",
    "Win cash prizes worth Rs 50,000! Reply WIN to enter.",
    "Hey, are we still meeting for the Python lab at 2 PM?",
    "Please send me the notes from today's Operating Systems lecture.",
    "Mom called, she asked you to pick up groceries on your way home."
]
labels = [1, 1, 1, 0, 0, 0]

# 2. Convert text messages into numerical word-count features
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(messages)

# 3. Train a Multinomial Naive Bayes machine learning model
model = MultinomialNB()
model.fit(X_train, labels)

# 4. Test the model on new unseen messages
test_messages = [
    "Claim your free lottery reward now!",
    "Can you share the GitHub link for our assignment?"
]
X_test = vectorizer.transform(test_messages)
predictions = model.predict(X_test)

for msg, pred in zip(test_messages, predictions):
    status = "SPAM" if pred == 1 else "NOT SPAM"
    print(f"Message: '{msg}' --> Prediction: {status}")
