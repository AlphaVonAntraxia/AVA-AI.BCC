import openai
import os

# Set your OpenAI API Key securely
openai.api_key = os.getenv("OPENAI_API_KEY")

def predict_with_llm(symbol, recent_data):
    """
    Generates a reasoning-based prediction using ChatGPT.
    """
    prompt = f"""You're an expert financial assistant. Based on the recent price trend of {symbol}, suggest the likely price direction for tomorrow.
Recent closing prices:
{recent_data.to_string(index=False)}

What do you predict the price will be tomorrow (in USD)? Just respond with the predicted price.
"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.7
        )
        output = response['choices'][0]['message']['content'].strip()
        return float(output.replace("$", "").replace(",", ""))
    except Exception as e:
        print("LLM prediction error:", e)
        return None
