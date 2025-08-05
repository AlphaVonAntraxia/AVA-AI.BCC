import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import ta

def add_features(data):
    data = data.copy()
    close = data['Close']
    if isinstance(close, pd.DataFrame) or len(close.shape) > 1:
        close = close.squeeze()  # Ensures it's a 1D Series

    data['RSI'] = ta.momentum.RSIIndicator(close).rsi()
    data['MACD'] = ta.trend.MACD(close).macd()
    data['SMA_10'] = close.rolling(window=10).mean()
    data['EMA_10'] = close.ewm(span=10, adjust=False).mean()
    return data

def create_lag_features(data, n_lags=5):
    for i in range(1, n_lags + 1):
        data[f'lag_{i}'] = data['Close'].shift(i)
    return data

def predict_with_xgboost(data):
    df = data.copy()
    df = add_features(df)
    df = create_lag_features(df, n_lags=5)
    df.dropna(inplace=True)

    features = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_5', 'RSI', 'MACD', 'SMA_10', 'EMA_10']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df[features])
    y = df['Close'].values

    split_index = int(len(df) * 0.8)
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    df.loc[df.index[split_index:], 'Predicted'] = y_pred

    # Predict next day's price
    latest_row = df[features].iloc[-1:].values
    latest_scaled = scaler.transform(latest_row)
    next_day_pred = model.predict(latest_scaled)[0]

    # Metrics (optional debug)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    print(f"[XGBoost] MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    return df[['Date', 'Close', 'Predicted']], round(float(next_day_pred), 2)
