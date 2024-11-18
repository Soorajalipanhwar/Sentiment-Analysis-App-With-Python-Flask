from flask import Flask, render_template, request
import pickle

# Load the trained model and vectorizer
with open('models/sentiment_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('models/vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    sentiment_class = None

    if request.method == "POST":
        # Get the user input from the form
        user_input = request.form["message"]

        # Vectorize the input text
        user_input_vectorized = tfidf.transform([user_input])

        # Predict sentiment: 0 = Negative, 1 = Positive
        prediction = model.predict(user_input_vectorized)[0]

        # Assign sentiment label and class
        sentiment = "Positive" if prediction == 1 else "Negative"
        sentiment_class = "positive" if prediction == 1 else "negative"

    return render_template("index.html", sentiment=sentiment, sentiment_class=sentiment_class)

if __name__ == "__main__":
    app.run(debug=True)
