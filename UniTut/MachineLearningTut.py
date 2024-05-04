import pandas as pd
from sklearn.metrics import classification_report, f1_score


# Define a function to categorize VADER sentiment
def categorize_sentiment(compound_score):
    if compound_score >= 0.05:
        return 2  # Positive
    elif compound_score <= -0.05:
        return 0  # Negative
    else:
        return 1  # Neutral


# Define a function to categorize AFINN sentiment
def categorize_afinn_sentiment(score):
    if score > 0:
        return 2  # Positive
    elif score < 0:
        return 0  # Negative
    else:
        return 1  # Neutral


# Define a function to calculate evaluation metrics
def calculate_metrics(df, labeled_data, method):
    # Merge data and labeled_data on a common identifier
    merged_data = pd.merge(df, labeled_data, left_index=True, right_index=True)

    # Drop rows with NaN values in the ground truth labels
    merged_data.dropna(subset=['GroundTruth'], inplace=True)

    # Generate classification report
    report = classification_report(merged_data['GroundTruth'], merged_data[method + ' Sentiment'],
                                   labels=classes, output_dict=True)

    precision_scores = {}
    recall_scores = {}
    f1_scores = {}

    # Extract precision, recall, and F1 scores for each class
    for label in classes:
        label_str = str(label)
        precision_scores[f'{method} Precision for {label_str}'] = report[label_str]['precision']
        recall_scores[f'{method} Recall for {label_str}'] = report[label_str]['recall']
        f1_scores[f'{method} F1 Score for {label_str}'] = report[label_str]['f1-score']

    # Calculate micro-average and macro-average F1 scores
    micro_f1 = f1_score(merged_data['GroundTruth'], merged_data[method + ' Sentiment'], average='micro', labels=classes)
    macro_f1 = f1_score(merged_data['GroundTruth'], merged_data[method + ' Sentiment'], average='macro', labels=classes)

    micro_f1_scores = {f'{method} Micro F1': micro_f1}
    macro_f1_scores = {f'{method} Macro F1': macro_f1}

    return precision_scores, recall_scores, f1_scores, micro_f1_scores, macro_f1_scores


# Define file paths for results and labeled data
file_path_results = r'C:\Users\Joe\PycharmProjects\pythonProject\Datasets\sentiment_analysis_results.csv'
file_path_labels = r'C:\Users\Joe\PycharmProjects\pythonProject\Datasets\youtube_comments_labels.csv'

# Read data from CSV files
data = pd.read_csv(file_path_results)
labeled_data = pd.read_csv(file_path_labels)

# Convert sentiment labels to numerical values for VADER, AFINN, and TextBlob
data['VADER Sentiment'] = data['VADER Compound'].apply(categorize_sentiment)
data['AFINN Sentiment'] = data['AFINN Score'].apply(categorize_afinn_sentiment)
data['TextBlob Sentiment'] = data['TextBlob Polarity'].apply(categorize_sentiment)

# Map sentiment labels to numerical values for ground truth
sentiment_mapping = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
labeled_data['GroundTruth'] = labeled_data['Sentiment'].map(sentiment_mapping)

# Initialize sentiment methods and classes
sentiment_methods = ['VADER', 'TextBlob', 'AFINN']
classes = [0, 1, 2]  # 0: Negative, 1: Neutral, 2: Positive

# Calculate and print evaluation metrics for each sentiment analysis method
for method in sentiment_methods:
    precision_scores, recall_scores, f1_scores, micro_f1_scores, macro_f1_scores = calculate_metrics(data, labeled_data,
                                                                                                     method)
    print(f"Metrics for {method}:")
    for label in classes:
        label_str = str(label)
        print(f"{method} Precision for {label_str}: {precision_scores[f'{method} Precision for {label_str}']}")
        print(f"{method} Recall for {label_str}: {recall_scores[f'{method} Recall for {label_str}']}")
        print(f"{method} F1 Score for {label_str}: {f1_scores[f'{method} F1 Score for {label_str}']}")
    print(f"{method} Micro F1: {micro_f1_scores[f'{method} Micro F1']}")
    print(f"{method} Macro F1: {macro_f1_scores[f'{method} Macro F1']}")
