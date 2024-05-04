import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import lime
import lime.lime_tabular
import webbrowser
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt

df = pd.read_csv("../Datasets/data.csv")


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
    print("---- Classification ----")
    print("Classification Accuracy:", accuracy)

    # Explainable AI (XAI) with Lime
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train.values, feature_names=X_train.columns.tolist(),
                                                       class_names=y.unique(), discretize_continuous=True)
    explanation = explainer.explain_instance(X_test.values[0], clf.predict_proba, num_features=2)

    # Show the explanation in HTML format
    html = explanation.as_html()

    # Save the explanation to a temporary HTML file
    with NamedTemporaryFile(suffix=".html", delete=False) as f:
        f.write(html.encode("utf-8"))
        temp_file = f.name

    # Open the temporary HTML file in the default web browser
    webbrowser.open("file://" + temp_file)


def clustering():
    print("\n---- Clustering ----")
    X = df[['Weight', 'Volume']]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X_scaled)
    labels = kmeans.labels_

    # Visualize the clusters
    plt.scatter(X['Weight'], X['Volume'], c=labels, cmap='viridis')
    plt.xlabel('Weight')
    plt.ylabel('Volume')
    plt.title('Clustering of Cars by Weight and Volume')
    plt.show()


def main():
    classification()
    clustering()


if __name__ == '__main__':
    main()
