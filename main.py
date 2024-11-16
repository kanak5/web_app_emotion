from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Initialize the FastAPI app
app = FastAPI()

# Load the trained model and vectorizer
vectorizer = joblib.load('vectorizer.pkl')
Classifier = joblib.load('logistic_model.pkl')

# Define input data structure
class InputText(BaseModel):
    text: str

# Define a text cleaning and processing function
def preprocess_text(text):
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    import string
    nltk.download("stopwords")
    nltk.download("WordNetLemmatizer")

    # Cleaning text
    text = text.lower()
    punc = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(punc)
    text = re.sub(r'\d+', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('\n', '', text)

    # Processing text
    stop_words = set(stopwords.words("english")) - set(["not"])
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]

    return " ".join(tokens)

# Define the prediction endpoint
@app.post("/predict-emotion/")
def predict_emotion(input_data: InputText):
    # Clean and preprocess the text
    cleaned_text = preprocess_text(input_data.text)

    # Transform the text into numerical features
    vectorized_text = vectorizer.transform([cleaned_text])

    # Predict the emotion
    predicted_label = Classifier.predict(vectorized_text)[0]
    label_mapping = {0: 'negative', 1: 'neutral', 2: 'positive'}
    predicted_emotion = label_mapping[predicted_label]
    print(predict_emotion)

    # Return the emotion
    return {"emotion": predicted_emotion}