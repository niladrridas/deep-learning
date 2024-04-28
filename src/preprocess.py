import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)

    # Convert 'Date' column to datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Define the features to be used in the model
    features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    df[features] = scaler.fit_transform(df[features])

    return df

if __name__ == "__main__":
    # Example usage
    file_path = 'data/TSLA.csv'
    preprocessed_df = preprocess_data(file_path)
    print(preprocessed_df.head())
