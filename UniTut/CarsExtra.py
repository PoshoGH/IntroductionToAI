import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

# Load the dataset
file_path = "../Datasets/data.csv"
df = pd.read_csv(file_path)

# Classification
def classification():
    X = df[['Weight', 'Volume']]
    y = pd.cut(df['CO2'], bins=3, labels=['Low', 'Medium', 'High'])

    # Split the dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the decision tree classifier
    clf = DecisionTreeClassifier(random_state=42)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = clf.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Classification Accuracy:", accuracy)

    # Explainable AI (XAI)
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns.tolist(),
                                                       class_names=y.unique(), discretize_continuous=True)
    explanation = explainer.explain_instance(X_test.values[0], clf.predict_proba, num_features=2)
    explanation.show_in_notebook()

# Sentiment Analysis
def sentiment_analysis():
    # Check if 'Comments' column exists in the dataframe
    if 'Comments' not in df.columns:
        print("Error: 'Comments' column not found in the dataset.")
        return

    comments = df['Comments'].fillna('')
    y = pd.cut(df['CO2'], bins=3, labels=['Low', 'Medium', 'High'])

    # Preprocess comments using CountVectorizer
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(comments, y)

    # Example sentiment analysis
    test_comments = ['This car is great!', 'The emissions are too high.']
    predictions = model.predict(test_comments)
    print("Sentiment Analysis Predictions:", predictions)


# Clustering
def clustering():
    X = df[['Weight', 'Volume']]

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)

    # Visualize clusters
    plt.scatter(X['Weight'], X['Volume'], c=kmeans.labels_, cmap='viridis')
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='*', s=300, c='red', label='Centroids')
    plt.xlabel('Weight')
    plt.ylabel('Volume')
    plt.title('Vehicle Clustering')
    plt.legend()
    plt.show()

def main():
    print("---- Classification ----")
    classification()

    print("\n---- Sentiment Analysis ----")
    sentiment_analysis()

    print("\n---- Clustering ----")
    clustering()

if __name__ == '__main__':
    main()
