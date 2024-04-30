import os
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import pandas as pd
import langid
import re
from afinn import Afinn  # Import AFINN
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import nltk


# Set up YouTube Data API
api_key = "AIzaSyA9ZupGnwLfLM4m9pwc-r202hkLhP77OA0"
youtube = build("youtube", "v3", developerKey=api_key)


# Define the YouTube video ID for which you want to analyze comments
video_id = 'ap1JDDnQIA8'


# Function to fetch all English comments from the YouTube video
def get_all_english_video_comments(youtube, **kwargs):
    comments = []
    timestamps = []  # Store timestamps
    while True:
        results = youtube.commentThreads().list(**kwargs).execute()
        for item in results['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            timestamp = item['snippet']['topLevelComment']['snippet']['publishedAt']
            lang, _ = langid.classify(comment)
            if lang == 'en':
                comments.append(comment)
                timestamps.append(timestamp)
        if 'nextPageToken' in results:
            kwargs['pageToken'] = results['nextPageToken']
        else:
            break
    return comments, timestamps


# Preprocess comments by removing special characters and lowercasing
def preprocess_comments(comments):
    preprocessed_comments = []
    for comment in comments:
        comment = re.sub(r'[^A-Za-z0-9 ]+', '', comment).lower()
        preprocessed_comments.append(comment)
    return preprocessed_comments


# Get all English comments and timestamps from the YouTube video
comments, timestamps = get_all_english_video_comments(youtube, part='snippet', videoId=video_id, textFormat='plainText')


# Preprocess comments
preprocessed_comments = preprocess_comments(comments)


# Sentiment analysis using VADER
vader_analyzer = SentimentIntensityAnalyzer()
vader_sentiments = [vader_analyzer.polarity_scores(comment) for comment in preprocessed_comments]


# Sentiment analysis using AFINN
afinn = Afinn()
afinn_sentiments = [afinn.score(comment) for comment in preprocessed_comments]


# Sentiment analysis using TextBlob
textblob_sentiments = [TextBlob(comment).sentiment.polarity for comment in preprocessed_comments]


# Create a DataFrame for visualization
data = {
    'Timestamps': timestamps,
    'Comments': comments,
    'VADER Compound': [s['compound'] for s in vader_sentiments],
    'AFINN Score': afinn_sentiments,
    'TextBlob Polarity': textblob_sentiments
}


df = pd.DataFrame(data)


# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=20)  # You can adjust max_features as needed


# Fit and transform the preprocessed comments
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_comments)


# Get the feature names (words)
feature_names = tfidf_vectorizer.get_feature_names_out()


# Calculate the mean TF-IDF score for each word across all comments
mean_tfidf_scores = np.mean(tfidf_matrix, axis=0).A1


# Create a DataFrame to store the words and their mean TF-IDF scores
tfidf_df = pd.DataFrame({'Word': feature_names, 'Mean TF-IDF Score': mean_tfidf_scores})


# Sort the DataFrame by mean TF-IDF score in descending order
tfidf_df = tfidf_df.sort_values(by='Mean TF-IDF Score', ascending=False)


# Print the top 20 words and their mean TF-IDF scores
print("Top 20 Words by Mean TF-IDF Score:")
print(tfidf_df[['Word', 'Mean TF-IDF Score']][:20])


def categorize_sentiment(compound_score):
    if compound_score >= 0.05:
        return 'Positive'
    elif compound_score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'


# Categorize sentiments for VADER, TextBlob, and AFINN
df['VADER Sentiment'] = df['VADER Compound'].apply(categorize_sentiment)
df['AFINN Sentiment'] = df['AFINN Score'].apply(categorize_sentiment)
df['TextBlob Sentiment'] = df['TextBlob Polarity'].apply(categorize_sentiment)


# Count the number of comments in each sentiment category for VADER
vader_sentiment_counts = df['VADER Sentiment'].value_counts()


# Count the number of comments in each sentiment category for AFINN
afinn_sentiment_counts = df['AFINN Sentiment'].value_counts()


# Count the number of comments in each sentiment category for TextBlob
textblob_sentiment_counts = df['TextBlob Sentiment'].value_counts()


# Print the count summary for VADER
print("VADER Sentiment Summary:")
print(vader_sentiment_counts)


# Print the count summary for AFINN
print("\nAFINN Sentiment Summary:")
print(afinn_sentiment_counts)


# Print the count summary for TextBlob
print("\nTextBlob Sentiment Summary:")
print(textblob_sentiment_counts)


# Time Series Plot for VADER, AFINN, and TextBlob
plt.figure(figsize=(12, 6))
plt.plot(df['Timestamps'], df['VADER Compound'], label='VADER', color='lightblue')
plt.plot(df['Timestamps'], afinn_sentiments, label='AFINN', color='lightcoral')
plt.plot(df['Timestamps'], textblob_sentiments, label='TextBlob', color='lightgreen')
plt.title(f'Sentiment Analysis Over Time for Video {video_id}')
plt.xlabel('Timestamp')
plt.ylabel('Sentiment Polarity Score')
plt.legend()
plt.xticks(rotation=45)


# Bar plot for sentiment comparison
plt.figure(figsize=(10, 6))
sentiments = ['Positive', 'Neutral', 'Negative']
vader_counts = [vader_sentiment_counts.get(sentiment, 0) for sentiment in sentiments]
afinn_counts = [afinn_sentiment_counts.get(sentiment, 0) for sentiment in sentiments]
textblob_counts = [textblob_sentiment_counts.get(sentiment, 0) for sentiment in sentiments]


width = 0.2
x = range(len(sentiments))


plt.bar(x, vader_counts, width, label='VADER', align='center')
plt.bar([i + width for i in x], afinn_counts, width, label='AFINN', align='center')
plt.bar([i + width * 2 for i in x], textblob_counts, width, label='TextBlob', align='center')


plt.xlabel('Sentiment Category')
plt.ylabel('Number of Comments')
plt.xticks([i + width for i in x], sentiments)
plt.title(f'Sentiment Analysis Comparison for Video {video_id}')
plt.legend()


# Show the top 20 words by mean TF-IDF score
plt.figure(figsize=(12, 6))
plt.barh(tfidf_df['Word'][:20], tfidf_df['Mean TF-IDF Score'][:20], color='lightblue')
plt.title('Top 20 Words by Mean TF-IDF Score')
plt.xlabel('Mean TF-IDF Score')
plt.ylabel('Word')
plt.gca().invert_yaxis()  # Invert the y-axis to show the highest scores at the top
plt.show()