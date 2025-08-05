# models/llm_model.py

def predict_with_llm(symbol, data):
    """
    Dummy LLM prediction logic. Replace this with real API call if needed.
    """
    import datetime
    import numpy as np

    last_price = data['Close'].iloc[-1]
    predicted_price = round(last_price * np.random.uniform(0.98, 1.02), 2)  # Fake small change

    prediction_date = datetime.datetime.now() + datetime.timedelta(days=1)
    return prediction_date.date(), predicted_price