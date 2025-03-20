import numpy as np
import matplotlib.pyplot as plt
import nltk
import pickle
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import requests
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc
from imblearn.over_sampling import RandomOverSampler

# Initialize necessary resources
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

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

# Function to fetch posts from Facebook and assign sentiment
def analyze_facebook_posts(user_id, access_token):
    url = f"https://graph.facebook.com/v15.0/{8539716629464572}/posts"
    params = {
        'access_token': "EAAJeKDEfAuMBO41WePlX2SjC4XRoTeOOJfOVC2me9boCDZCzTILnNEZBtQq2orf6Q54VwLcE1wzvkAHFOGC1tMpLtJ3bgFgvH2u27QkM7OOaZCL7VrjsCXLVTNU4d3szX0gotPtQvPzxPaOVMCqZBhHlordBvqdcbBpJummjpMv7qeGBBXbViWujNc1aMoZBu7iNNO3rbYoReSAH7ToKljdLneOTOXU3JfuCE0ZARLSFxhuycO2P5AVQZDZD",
        'fields': 'message',
        'limit': 100  # Fetch more posts
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    posts = []
    for post in data['data']:
        text = post.get('message', '')
        # Preprocess the text
        preprocessed_text = preprocess_text(text)

        # Sentiment analysis of the preprocessed text
        text_sentiment, text_sentiment_score = sentiment_analysis(preprocessed_text)
        
        # Handle emojis if available (assuming emoji extraction can be done from the text)
        emojis = [char for char in text if char in '😀😊😍😂😭😢😡😞']  # Extract emojis from the text
        emoji_sentiment_score = emoji_sentiment(emojis)
        
        # Combine both text sentiment and emoji sentiment (simple average approach)
        combined_sentiment_score = text_sentiment_score + emoji_sentiment_score

        # Determine the final sentiment
        if combined_sentiment_score > 0.1:
            final_sentiment = 'Positive'
        elif combined_sentiment_score < -0.5:
            final_sentiment = 'Negative'
        else:
            final_sentiment = 'Neutral'
        
        posts.append({'text': text, 'sentiment': final_sentiment})
    
    return posts

# Replace this with your actual Facebook user ID and access token
user_id = '8539716629464572'
access_token = "EAAJeKDEfAuMBO41WePlX2SjC4XRoTeOOJfOVC2me9boCDZCzTILnNEZBtQq2orf6Q54VwLcE1wzvkAHFOGC1tMpLtJ3bgFgvH2u27QkM7OOaZCL7VrjsCXLVTNU4d3szX0gotPtQvPzxPaOVMCqZBhHlordBvqdcbBpJummjpMv7qeGBBXbViWujNc1aMoZBu7iNNO3rbYoReSAH7ToKljdLneOTOXU3JfuCE0ZARLSFxhuycO2P5AVQZDZD"
posts_data = analyze_facebook_posts(user_id, access_token)

# Convert the data into a DataFrame
df = pd.DataFrame(posts_data)

# Display the first few rows of the DataFrame
print(df.head())

# Optional: Save the DataFrame to a CSV file for further inspection or use
df.to_csv('facebook_posts.csv', index=False)

# Preprocess and vectorize the text data
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['text'])  # Text features (vectorized)
y = df['sentiment'].map({'Positive': 1, 'Negative': 0, 'Neutral': 0})  # Sentiment labels (0 = negative, 1 = positive)

# Resampling (Oversampling) to balance the classes
oversampler = RandomOverSampler(sampling_strategy='minority', random_state=42)
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Split the resampled data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')

# Save the trained model and vectorizer to disk
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

print("Model and vectorizer saved!")

# Print the value counts for sentiment
print(df['sentiment'].value_counts())

# Confusion Matrix
labels = [0, 1]
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Print classification report
print(classification_report(y_test, y_pred))

y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calculate AUC score
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', linewidth=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
