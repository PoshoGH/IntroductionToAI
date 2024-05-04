import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    """
    Load the dataset from the given file path.

    Args:
        file_path (str): Path to the CSV file containing the dataset.

    Returns:
        pandas.DataFrame: Loaded DataFrame containing the dataset.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        print("Error loading data:", e)
        return None


def preprocess_data(df):
    """
    Preprocess the dataset by extracting features and target variable,
    and scaling the features using StandardScaler.

    Args:
        df (pandas.DataFrame): Input DataFrame containing the dataset.

    Returns:
        tuple: A tuple containing scaled features, target variable, and StandardScaler object.
    """
    try:
        # Extract features (Weight, Volume) and target variable (CO2)
        X = df[['Weight', 'Volume']]
        y = df['CO2']

        # Scale the features using StandardScaler
        scaler = StandardScaler()
        scaledX = scaler.fit_transform(X)

        return scaledX, y, scaler
    except Exception as e:
        print("Error preprocessing data:", e)
        return None, None, None


def train_model(X, y):
    """
    Train a linear regression model using the given features and target variable.

    Args:
        X (numpy.ndarray): Scaled features.
        y (pandas.Series): Target variable.

    Returns:
        sklearn.linear_model.LinearRegression: Trained linear regression model.
    """
    try:
        regr = LinearRegression()
        regr.fit(X, y)
        return regr
    except Exception as e:
        print("Error training model:", e)
        return None


def predict_CO2(model, scaler, weight, volume):
    """
    Predict CO2 emission given weight and volume using the trained model.

    Args:
        model (sklearn.linear_model.LinearRegression): Trained linear regression model.
        scaler (StandardScaler): StandardScaler object used for scaling features.
        weight (float): Weight of the car.
        volume (float): Volume of the car.

    Returns:
        float: Predicted CO2 emission.
    """
    try:
        # Scale the new data
        scaled = scaler.transform([[weight, volume]])

        # Predict CO2 emission
        predicted_CO2 = model.predict(scaled)
        return predicted_CO2[0]
    except Exception as e:
        print("Error predicting CO2:", e)
        return None


def main():
    # Load data
    file_path = "../Datasets/data.csv"
    df = load_data(file_path)
    if df is None:
        return

    # Preprocess data
    X, y, scaler = preprocess_data(df)
    if X is None or y is None or scaler is None:
        return

    # Train model
    model = train_model(X, y)
    if model is None:
        return

    # New data
    weight = 2300
    volume = 1.3

    # Predict CO2 emission
    predicted_CO2 = predict_CO2(model, scaler, weight, volume)
    if predicted_CO2 is not None:
        print("Predicted CO2:", predicted_CO2)


if __name__ == '__main__':
    main()
