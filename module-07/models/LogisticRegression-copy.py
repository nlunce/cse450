import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
from sklearn.pipeline import make_pipeline


import numpy as np

pd.set_option("display.max_columns", None)

df = pd.read_csv("../datasets/spam-email+sms-dataset.csv")


def preprocess(df):
    # Remove duplicates
    df = df.drop_duplicates()

    # Ensure 'message' is treated as a string
    df["message"] = df["message"].astype(str)
    df["message_type"] = df["message_type"].astype(str)

    # Add a column for message length
    df["message_length"] = df["message"].apply(len)

    # Add a column for word count
    df["word_count"] = df["message"].apply(lambda x: len(x.split()))

    # Create 'email' and 'sms' binary columns based on 'message_type'
    df["email"] = df["message_type"].apply(lambda x: 1 if x == "email" else 0)
    df["sms"] = df["message_type"].apply(lambda x: 1 if x == "sms" else 0)

    # Initialize CountVectorizer
    vectorizer = CountVectorizer()

    # Rearrange the columns - drop message_type
    df = df[["message", "email", "sms", "word_count", "message_length", "spam"]]

    return df


df = preprocess(df)


X = df.drop("spam", axis=1)
y = df[["spam"]]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Separate the 'message' column from other features
X_train_text = X_train["message"]
X_train_other_features = X_train.drop("message", axis=1)

X_test_text = X_test["message"]
X_test_other_features = X_test.drop("message", axis=1)

# Initialize and fit CountVectorizer on the training text data
vectorizer = CountVectorizer()
X_train_text_transformed = vectorizer.fit_transform(X_train_text)

# Transform the test text data using the fitted vectorizer
X_test_text_transformed = vectorizer.transform(X_test_text)

# Horizontally stack the other features with the transformed text features
X_train_final = hstack([X_train_text_transformed, X_train_other_features])
X_test_final = hstack([X_test_text_transformed, X_test_other_features])

# Train the model
model = LogisticRegression(random_state=42, max_iter=10000)
model.fit(X_train_final, y_train.values.ravel())

# Predict and evaluate
predictions = pipeline.predict(X_test_final)
score = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {score:.2f}")
