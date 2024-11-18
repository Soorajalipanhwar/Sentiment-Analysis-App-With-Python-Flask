import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle

# Load the dataset (Assuming your dataset is named 'sentiment_data.csv')
# Ensure that the CSV file contains columns: 'text' for the sentence and 'label' for the sentiment (0 = negative, 1 = positive)
df = pd.read_csv('movie.csv')

# Check if the dataset loaded correctly
print(f"Dataset loaded with {len(df)} rows.")
print(df.head())

# Prepare the data: X for features (text), y for target (sentiment label)
X = df['text']
y = df['label']

# Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data (convert the text into numerical features)
tfidf = TfidfVectorizer(stop_words='english')
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)  # Increase iterations if convergence warning occurs
model.fit(X_train_tfidf, y_train)

# Make predictions and evaluate the model
y_pred = model.predict(X_test_tfidf)

# Print classification report to evaluate accuracy, precision, recall, etc.
print(f"Model Accuracy: {model.score(X_test_tfidf, y_test) * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and the vectorizer to disk
with open('sentiment_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tfidf, vectorizer_file)

print("Model and vectorizer have been saved.")
