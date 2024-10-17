from flask import Flask, request, render_template
from pickle import load
import regex as re
import os
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer


# Define the Flask app and set the template folder path
app = Flask(__name__, template_folder='../templates')

# Get the absolute path to the model file
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../models/svm_classifier_C-1_deg-1_gam-scale_ker-linear_42.sav")
model = load(open(model_path, "rb"))

class_dict = {
    "0": "POSITIVE",
    "1": "NEGATIVE"
}


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
            text = re.sub("&lt;/?.*?&gt;"," &lt;&gt; ", text)

            return text.split()

        val1 = preprocess_text(val1)
        print(val1)

        download("wordnet")
        lemmatizer = WordNetLemmatizer()

        download("stopwords")
        stop_words = stopwords.words("english")

        def lemmatize_text(words, lemmatizer = lemmatizer):
            tokens = [lemmatizer.lemmatize(word) for word in words]
            tokens = [word for word in tokens if word not in stop_words]
            tokens = [word for word in tokens if len(word) > 3]
            return tokens

        val1 = lemmatize_text(val1)
        print(val1)
        tokens_list = val1
        tokens_list = [" ".join(val1)]
        print(tokens_list)

        vectorizer = TfidfVectorizer(max_features = 5000, max_df = 0.8, min_df = 5)
        val1 = vectorizer.fit_transform(tokens_list).toarray()
        val1[:5]
        #print(val1[:5])

        data = [[val1]] #CHANGED THIS
        print(data)
        #data = [val1]
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