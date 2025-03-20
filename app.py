import nltk
import os
import string
import joblib
import requests
from flask import Flask, render_template, request
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem import PorterStemmer
from nltk.tokenize import PunktTokenizer


#nltk.download('punkt')      # For tokenization
#nltk.download('stopwords')  # For stopword removal
#nltk.download('wordnet') 
#nltk.download('omw-1.4')

# Initialize the Flask application
app = Flask(__name__)

# Define file paths for the model and vectorizer
model_file_path = 'sentiment_model.pkl'
vectorizer_file_path = 'vectorizer.pkl'

# Initialize the lemmatizer for preprocessing
lemmatizer = WordNetLemmatizer()

# Load the pre-trained sentiment model and vectorizer
def load_model_and_vectorizer():
    # Check if the model files exist
    if not os.path.exists(model_file_path) or not os.path.exists(vectorizer_file_path):
        raise FileNotFoundError("Model or vectorizer files not found.")
    
    # Load the model and vectorizer
    model = joblib.load(model_file_path)
    vectorizer = joblib.load(vectorizer_file_path)
    
    return model, vectorizer

# Load the model and vectorizer at the start of the app
model, vectorizer = load_model_and_vectorizer()

# Function to preprocess text (cleaning, tokenization, stopword removal, lemmatization)
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove special characters, numbers, and punctuation
    text = re.sub(r'[^a-z\s]', '', text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    return ' '.join(lemmatized_tokens)

# Function to handle emojis and return sentiment score based on emojis
def emoji_sentiment(emojis):
    positive_emojis = ['😀', '😊', '😍', '😂']
    negative_emojis = ['😭', '😢', '😡', '😞']

    positive_count = sum([1 for e in emojis if e in positive_emojis])
    negative_count = sum([1 for e in emojis if e in negative_emojis])

    sentiment_score = positive_count - negative_count  # Positive for more positive emojis, negative for more negative emojis
    return sentiment_score

# Function for sentiment analysis using TextBlob
def sentiment_analysis(text):
    # Analyze the sentiment of the post
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity  # Range from -1 (negative) to 1 (positive)

    # Classify sentiment based on score
    if sentiment_score > 0.2:
        sentiment = 'Positive'
    elif sentiment_score < -0.2:
        sentiment = 'Negative'
    else:
        sentiment = 'Neutral'

    return sentiment, sentiment_score

# Function to simulate fetching Facebook posts
def get_facebook_posts(user_id, access_token):
    url = f"https://graph.facebook.com/{8539716629464572}/feed?access_token={"EAAJeKDEfAuMBOxX1zSFWIOifATLvTR42uum5aSPhCo9xnvMQKIG5NesJSkC7bsLohtntBpzP68KXS4t5F4vmV7MPmY1jzejMUULMnofMBKahhAba5ZCPU2f6rtbbZAy0asrZC0Mg5H32BlI9JleZAQqQp7yOCoGfBO0w57N9VSlckgvkkff57EWTOc0fN6UmIAdp6hrmq5AB5ItFHU54YSC7jU7FSrqpU9OZBPsJ4CZChZCYrq1ftR6"}&fields=message,id,comments"
    response = requests.get(url)
    
    # Check for errors in the response
    if response.status_code != 200:
        raise Exception(f"Failed to fetch posts. Error: {response.status_code}")
    
    posts_data = response.json()  # Parse the JSON response
    posts = []
    for post in posts_data.get('data', []):
        if 'message' in post:  # Only consider posts with a message
            posts.append({'text': post['message'], 'sentiment': None})
    
    return posts

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        access_token = request.form['access_token']
        user_id = request.form['user_id']
        
        try:
            # Get the Facebook posts using the provided user_id and access_token
            posts = get_facebook_posts(user_id, access_token)
            
            # Analyze sentiment of each post using the pre-trained model and vectorizer
            for post in posts:
                # Preprocess the text (this is the main change: preprocessing)
                preprocessed_text = preprocess_text(post['text'])
                
                # Analyze sentiment of the preprocessed text using TextBlob
                text_sentiment, text_sentiment_score = sentiment_analysis(preprocessed_text)

                # Now you can also analyze emojis using emoji sentiment
                emojis = [e for e in post['text'] if e in ['😀', '😊', '😍', '😂', '😭', '😢', '😡', '😞']]
                emoji_sentiment_score = emoji_sentiment(emojis)

                # Combine both text and emoji sentiment scores for final classification
                combined_sentiment_score = text_sentiment_score + emoji_sentiment_score

                # Classify based on the combined sentiment score
                if combined_sentiment_score > 0.2:
                    post['sentiment'] = 'Positive'
                elif combined_sentiment_score < -0.2:
                    post['sentiment'] = 'Negative'
                else:
                    post['sentiment'] = 'Neutral'
            
            # Render results on the results.html page
            return render_template('results.html', posts=posts)
        
        except Exception as e:
            # If any error occurs during fetching posts or analyzing sentiment
            return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    # Start the Flask app in debug mode
    app.run(debug=True)
