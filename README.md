# Tesla Stock Price Prediction using LSTM

This project uses a Long Short-Term Memory (LSTM) model, a type of Recurrent Neural Network (RNN), to predict the future stock prices of Tesla (TSLA) based on historical data.

## Libraries Used

- numpy
- pandas
- sklearn
- tensorflow
- matplotlib

## How to Run

1. Ensure that you have the necessary libraries installed. You can install them with pip:

```bash
pip install numpy pandas sklearn tensorflow matplotlib
```

2. Run the `main.py` script:

```bash
python main.py
```

### Project Structure
The `main.py` script loads the data from a CSV file named 'TSLA.csv' into a pandas DataFrame, prepares the data for LSTM, scales the data, splits the data into training and testing sets, builds the LSTM model, compiles and trains the model, plots the training and validation loss, makes predictions on the test data, inverse scales the predicted and actual values, and calculates the Root Mean Squared Error (RMSE) between the predicted and actual values.

### Results
The model's prediction error is measured by the RMSE, which is printed at the end of the script. The training and validation loss is also plotted against the number of epochs to visualize the model's learning process.


![Figure_1](https://github.com/niladrridas/deep-learning/blob/main/data/Figure_1.png)

![Figure_2](https://github.com/niladrridas/deep-learning/blob/main/data/Figure_2.png)
