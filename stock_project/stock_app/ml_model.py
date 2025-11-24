import yfinance as yf
import joblib
import os
import pandas as pd
from sklearn.linear_model import LinearRegression

def train_model(stock_name):
    data = yf.download(stock_name, period="6mo", interval="1d")

    if data is None or data.empty:
        raise ValueError("No data found for stock: " + stock_name)

    data = data.dropna()

    X = data.index.factorize()[0].reshape(-1, 1)
    y = data["Close"].values

    model = LinearRegression()
    model.fit(X, y)

    os.makedirs("ml_models", exist_ok=True)
    joblib.dump(model, f"ml_models/{stock_name}_model.pkl")


def predict_price(stock_name):
    # Try loading model first
    model_path = f"ml_models/{stock_name}_model.pkl"

    if not os.path.exists(model_path):
        train_model(stock_name)

    model = joblib.load(model_path)

    # Get new data
    data = yf.download(stock_name, period="6mo", interval="1d")

    if data is None or data.empty:
        raise ValueError("No data available to predict.")

    data = data.dropna()

    last_index = data.index.factorize()[0][-1]

    last_close = data["Close"].iloc[-1]

    predicted_price = model.predict([[last_index]])[0]

    return predicted_price
