# Import necessary libraries
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from afinn import Afinn
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
import langid
from googleapiclient.discovery import build

# Set up YouTube Data API
api_key = "AIzaSyA9ZupGnwLfLM4m9pwc-r202hkLhP77OA0"  # YouTube Data API key
youtube = build("youtube", "v3", developerKey=api_key)

# Define the YouTube video ID for which you want to analyze comments
video_id = 'ap1JDDnQIA8'


# Function to fetch all English comments from the YouTube video
def get_all_english_video_comments(youtube, **kwargs):
    """
    Fetch all English comments and their timestamps from a YouTube video.

    Args:
        YouTube: YouTube Data API object.
        **kwargs: Keyword arguments for the commentThreads. List method.

    Returns:
        comments: List of English comments.
        timestamps: List of timestamps corresponding to the comments.
    """
    comments = []
    timestamps = []
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
    """
    Preprocess comments by removing special characters and converting to lowercase.

    Args:
        comments: List of comments.

    Returns:
        preprocessed_comments: List of preprocessed comments.
    """
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
vader_sentiments = [vader_analyzer.polarity_scores(comment)['compound'] for comment in preprocessed_comments]

# Sentiment analysis using AFINN
afinn = Afinn()
afinn_sentiments = [afinn.score(comment) for comment in preprocessed_comments]

# Sentiment analysis using TextBlob
textblob_sentiments = [TextBlob(comment).sentiment.polarity for comment in preprocessed_comments]

# Create a DataFrame for visualization
data = {
    'Timestamps': timestamps,
    'Comments': comments,
    'VADER Compound': vader_sentiments,
    'AFINN Score': afinn_sentiments,
    'TextBlob Polarity': textblob_sentiments
}

df = pd.DataFrame(data)


# Categorize sentiments
def categorize_sentiment(compound_score):
    """
    Categorize sentiment based on compound score.

    Args:
        compound_score: Sentiment compound score.

    Returns:
        sentiment_category: Categorized sentiment (Positive, Neutral, Negative).
    """
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

# Count sentiment categories
vader_sentiment_counts = df['VADER Sentiment'].value_counts()
afinn_sentiment_counts = df['AFINN Sentiment'].value_counts()
textblob_sentiment_counts = df['TextBlob Sentiment'].value_counts()

# Plot sentiment over time
plt.figure(figsize=(12, 6))
plt.plot(df['Timestamps'], df['VADER Compound'], label='VADER', color='lightblue')
plt.plot(df['Timestamps'], afinn_sentiments, label='AFINN', color='lightcoral')
plt.plot(df['Timestamps'], textblob_sentiments, label='TextBlob', color='lightgreen')
plt.title(f'Sentiment Analysis Over Time for Video {video_id}')
plt.xlabel('Timestamp')
plt.ylabel('Sentiment Polarity Score')
plt.legend()
plt.xticks(rotation=45)
plt.show()

# Plot sentiment comparison
plt.figure(figsize=(10, 6))
sentiments = ['Positive', 'Neutral', 'Negative']
vader_counts = [vader_sentiment_counts.get(sentiment, 0) for sentiment in sentiments]
afinn_counts = [afinn_sentiment_counts.get(sentiment, 0) for sentiment in sentiments]
textblob_counts = [textblob_sentiment_counts.get(sentiment, 0) for sentiment in sentiments]

width = 0.2
x = np.arange(len(sentiments))

plt.bar(x, vader_counts, width, label='VADER', align='center')
plt.bar(x + width, afinn_counts, width, label='AFINN', align='center')
plt.bar(x + 2 * width, textblob_counts, width, label='TextBlob', align='center')

plt.xlabel('Sentiment Category')
plt.ylabel('Number of Comments')
plt.xticks(x + width, sentiments)
plt.title(f'Sentiment Analysis Comparison for Video {video_id}')
plt.legend()
plt.show()

# TF-IDF analysis
tfidf_vectorizer = TfidfVectorizer(max_features=20)
tfidf_matrix = tfidf_vectorizer.fit_transform(preprocessed_comments)
feature_names = tfidf_vectorizer.get_feature_names_out()
mean_tfidf_scores = np.mean(tfidf_matrix, axis=0).A1
tfidf_df = pd.DataFrame({'Word': feature_names, 'Mean TF-IDF Score': mean_tfidf_scores})
tfidf_df = tfidf_df.sort_values(by='Mean TF-IDF Score', ascending=False)

plt.figure(figsize=(12, 6))
plt.barh(tfidf_df['Word'][:20], tfidf_df['Mean TF-IDF Score'][:20], color='lightblue')
plt.title('Top 20 Words by Mean TF-IDF Score')
plt.xlabel('Mean TF-IDF Score')
plt.ylabel('Word')
plt.gca().invert_yaxis()
plt.show()
