import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta

def predict_with_catboost(data):
    df = data.copy()

    # --- Feature Engineering ---
    df['High_Low'] = df['High'] - df['Low']
    df['Open_Close'] = df['Open'] - df['Close']
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['Volume_Change'] = df['Volume'].pct_change()
    df['Return'] = df['Close'].pct_change()

    df = df.dropna().reset_index(drop=True)

    # --- Lag Features ---
    for lag in range(1, 6):
        df[f'lag_{lag}'] = df['Close'].shift(lag)

    df = df.dropna().reset_index(drop=True)

    # --- Define features and target ---
    features = ['Open', 'High', 'Low', 'Volume', 'High_Low', 'Open_Close',
                'MA10', 'MA20', 'Volume_Change', 'Return'] + [f'lag_{i}' for i in range(1, 6)]

    target = 'Close'

    X = df[features]
    y = df[target]

    # --- Normalize Features ---
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Train/Test Split ---
    X_train, X_test = X_scaled[:-1], X_scaled[-1:]
    y_train = y[:-1]

    # --- Train CatBoost Regressor ---
    model = CatBoostRegressor(verbose=0)
    model.fit(X_train, y_train)

    # --- Make Predictions for Entire Dataset ---
    df['Predicted'] = model.predict(X_scaled)

    # --- Predict for Next Day ---
    next_day_pred = model.predict(X_test)[0]
    next_date = df['Date'].iloc[-1] + timedelta(days=1)

    df_next = pd.DataFrame({
        'Date': [next_date],
        'Close': [np.nan],
        'Predicted': [next_day_pred]
    })

    df = pd.concat([df, df_next], ignore_index=True)

    return df[['Date', 'Close', 'Predicted']], round(float(next_day_pred), 2)
