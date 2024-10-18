from flask import Flask, request, render_template
from pickle import load
import regex as re
import os
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#website link https://ml-web-app-using-flask-e8bg.onrender.com/


# Define the Flask app and set the template folder path
app = Flask(__name__, template_folder='../templates')

# Get the absolute path to the model file
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/svm_classifier_C-1_deg-1_gam-scale_ker-linear_42.sav")
model = load(open(model_path, "rb"))

# Load the pre-fitted vectorizer
vectorizer_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/tfidf_vectorizer.sav")
vectorizer = load(open(vectorizer_path, "rb"))

class_dict = {
    "1": "POSITIVE",
    "0": "NEGATIVE"
}

# Ensure wordnet and stopwords are downloaded once
download("wordnet")
download("stopwords")
stop_words = stopwords.words("english")
lemmatizer = WordNetLemmatizer()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Handle form submission
        val1 = str(request.form["val1"])

        def preprocess_text(text):
            # Remove any character that is not a letter (a-z) or white space ( )
            text = re.sub(r'[^a-z ]', " ", text)

            # Remove white spaces
            text = re.sub(r'\s+[a-zA-Z]\s+', " ", text)
            text = re.sub(r'\^[a-zA-Z]\s+', " ", text)

            # Multiple white spaces into one
            text = re.sub(r'\s+', " ", text.lower())

            # Remove tags
            text = re.sub("&lt;/?.*?&gt;", " &lt;&gt; ", text)

            return text.split()

        val1 = preprocess_text(val1)

        def lemmatize_text(words, lemmatizer=lemmatizer):
            tokens = [lemmatizer.lemmatize(word) for word in words]
            tokens = [word for word in tokens if word not in stop_words]
            tokens = [word for word in tokens if len(word) > 3]
            return tokens

        val1 = lemmatize_text(val1)

        # Join tokens back into a single string for vectorizer
        tokens_list = [" ".join(val1)]

        # Use the pre-trained vectorizer to transform the input
        val1 = vectorizer.transform(tokens_list).toarray()

        # Ensure the correct shape for model input
        data = val1

        # Make a prediction using the SVM model
        prediction = str(model.predict(data)[0])
        pred_class = class_dict[prediction]
    else:
        # Handle initial GET request
        pred_class = None

    # Render the template with the prediction result (or None if GET request)
    return render_template("index.html", prediction=pred_class)


if __name__ == "__main__":
    # Use the port provided by Render, or default to 5000
    port = int(os.environ.get("PORT", 5000))
    # Set host to 0.0.0.0 to be accessible externally
    app.run(host="0.0.0.0", port=port, debug=True)
