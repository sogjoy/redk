from flask import Flask, request, render_template
import yfinance as yf
import pandas as pd
import numpy as np
import openai
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os

load_dotenv()
app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

def calculate_vader_indicators(df):
    # Calculate VADER components
    length = 10
    der_avg = 5
    smooth = 3
    
    # Volume calculations
    df['RelativeVolume'] = (df['Volume'] - df['Volume'].rolling(20).min()) / \
                          (df['Volume'].rolling(20).max() - df['Volume'].rolling(20).min())
    
    # Calculate 2-bar range
    df['HighestHigh'] = df['High'].rolling(2).max()
    df['LowestLow'] = df['Low'].rolling(2).min()
    df['R'] = (df['HighestHigh'] - df['LowestLow']) / 2
    
    # Calculate directional energy
    df['PriceChange'] = df['Close'].diff()
    df['SR'] = df['PriceChange'] / df['R']
    df['RSR'] = df['SR'].clip(-1, 1)
    df['C'] = df['RSR'] * df['RelativeVolume']
    
    # Calculate energy components
    df['C_Plus'] = df['C'].clip(lower=0)
    df['C_Minus'] = (-df['C']).clip(lower=0)
    
    # Calculate final indicators
    df['ADP'] = df['C_Plus'].rolling(length).mean() * 100
    df['ASP'] = df['C_Minus'].rolling(length).mean() * 100
    df['ANP'] = df['ADP'] - df['ASP']
    df['ANP_S'] = df['ANP'].rolling(smooth).mean()
    
    return df

def analyze_with_openai(vader_data):
    prompt = f"""
    Analyze the following VADER indicator values for trading decisions:
    
    ADP (Demand Energy): {vader_data['ADP']:.2f}
    ASP (Supply Energy): {vader_data['ASP']:.2f}
    ANP (Net Energy): {vader_data['ANP']:.2f}
    ANP_S (Smoothed Net Energy): {vader_data['ANP_S']:.2f}
    
    Based on these values, provide:
    1. Current market sentiment
    2. Trading recommendation (Buy/Sell/Hold)
    3. Key levels to watch
    4. Risk management advice
    """
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a professional trading analyst specializing in VADER indicator analysis."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        symbol = request.form['symbol']
        try:
            # Fetch data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)
            stock = yf.download(symbol, start=start_date, end=end_date)
            
            # Calculate VADER indicators
            vader_df = calculate_vader_indicators(stock)
            latest_data = vader_df.iloc[-1]
            
            # Get AI analysis
            analysis = analyze_with_openai(latest_data)
            
            result = {
                'symbol': symbol,
                'vader_indicators': {
                    'ADP': latest_data['ADP'],
                    'ASP': latest_data['ASP'],
                    'ANP': latest_data['ANP'],
                    'ANP_S': latest_data['ANP_S']
                },
                'analysis': analysis,
                'price': latest_data['Close']
            }
            
            return render_template('index.html', result=result)
            
        except Exception as e:
            return render_template('index.html', error=str(e))
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True) 