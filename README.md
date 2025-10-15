# Stock-price-prediction-project
Minor Project: Stock Price Prediction using LSTM

1. Introduction

                     Stock prices are influenced by several factors like demand, supply, economy, and investor behavior. Predicting them accurately is an important task in finance. This project uses Long Short-Term Memory (LSTM), a deep learning model, to predict future stock prices based on past historical data.

2. Objective

• Analyze historical stock data and extract useful patterns.
• Train an LSTM model to predict future stock closing prices.
• Visualize both historical and predicted prices.


3. Tools and Technologies

Programming Language: Python
Libraries Used: Streamlit, yfinance, pandas, numpy, sklearn, tensorflow, plotly, matplotlib
Algorithm Used: Long Short-Term Memory (LSTM) Neural Network

4. Dataset Description

• Source: Yahoo Finance (using yfinance library)
• Features: Date, Open, High, Low, Close, Volume
• Target: Close Price (for prediction)
• Duration: From 2015-01-01 to current date
5. Methodology

•	Step 1: Data Collection

•	Stock data is fetched using the yfinance API for a given symbol (e.g., AAPL, TSLA).
•	Step 2: Data Pre-processing
•	• Extract the 'Close' price column.
•	• Normalize using MinMaxScaler.
•	• Create 60-day sequences as LSTM inputs.
•	Step 3: Model Building (LSTM)
•	• Two stacked LSTM layers (50 units each).
•	• One Dense output layer.
•	• Optimizer: Adam | Loss: Mean Squared Error
•	Step 4: Model Training
•	Train model for 10 epochs with batch size of 32.
•	Step 5: Future Prediction
•	Predict the next 30 days of stock prices using the trained model.
•	Step 6: Visualization
•	Visualize both historical and predicted prices using matplotlib and Plotly.



6. Coding
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import plotly.graph_objects as go

st.title("📈 Stock Price Prediction App (LSTM)")

# --- Input ---
symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, TSLA, INFY)", "AAPL")
st.write(f"Fetching data for: {symbol}")

# --- Fetch Data ---
data = yf.download(symbol, start="2015-01-01", end=pd.Timestamp.today().strftime("%Y-%m-%d"))
st.subheader("Historical Stock Prices")
st.write(data.tail())

# --- Preprocessing ---
close_prices = data['Close'].values.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_prices)

# Create training data
sequence_length = 60
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i,0])
    y.append(scaled_data[i,0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# --- Build LSTM Model ---
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
st.write("Training LSTM model... this may take a minute ⏳")
model.fit(X, y, epochs=10, batch_size=32, verbose=0)
st.success("Model trained ✅")

# --- Predict next 30 days ---
last_sequence = scaled_data[-sequence_length:]
future_preds = []
current_seq = last_sequence

for _ in range(30):
    current_seq_reshaped = current_seq.reshape((1, sequence_length, 1))
    pred = model.predict(current_seq_reshaped, verbose=0)
    future_preds.append(pred[0,0])
    current_seq = np.append(current_seq[1:], pred, axis=0)

future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1,1))

# --- Display Prediction ---
future_dates = pd.date_range(start=data.index[-1]+pd.Timedelta(days=1), periods=30)
pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_preds.flatten()})
st.subheader("Next 30 Days Predicted Prices")
st.write(pred_df)

# --- Plot ---
fig = go.Figure()
fig.add_trace(go.Scatter(x=data.index, y=data['Close'], name='Historical Close'))
fig.add_trace(go.Scatter(x=pred_df['Date'], y=pred_df['Predicted Close'], name='Predicted Close'))
fig.update_layout(title=f"{symbol} Stock Price Prediction", xaxis_title="Date", yaxis_title="Price")
st.plotly_chart(fig)


7. Modules and Explanation

# =========================
# 1️⃣ Import Libraries
# =========================
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from datetime import timedelta

# =========================
# 2️⃣ User Input
# =========================
symbol = input("Enter Stock Symbol (e.g., AAPL, TSLA, INFY): ")
print(f"Fetching data for: {symbol}")

# =========================
# 3️⃣ Fetch Historical Data
# =========================
data = yf.download(symbol, start="2015-01-01", end=pd.Timestamp.today().strftime("%Y-%m-%d"))
print("Last 5 rows of historical data:")
print(data.tail())

# =========================
# 4️⃣ Preprocess Data
# =========================
close_prices = data['Close'].values.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_prices)

sequence_length = 60
X, y = [], []
for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i,0])
    y.append(scaled_data[i,0])

X, y = np.array(X), np.array(y)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# =========================
# 5️⃣ Build and Train LSTM Model
# =========================
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
print("Training LSTM model... this may take a few minutes ⏳")
model.fit(X, y, epochs=10, batch_size=32, verbose=1)
print("Model trained ✅")

# =========================
# 6️⃣ Predict Next 30 Days
# =========================
last_sequence = scaled_data[-sequence_length:]
future_preds = []
current_seq = last_sequence

for _ in range(30):
    current_seq_reshaped = current_seq.reshape((1, sequence_length, 1))
    pred = model.predict(current_seq_reshaped, verbose=0)
    future_preds.append(pred[0,0])
    current_seq = np.append(current_seq[1:], pred, axis=0)

future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1,1))
future_dates = pd.date_range(start=data.index[-1]+timedelta(days=1), periods=30)
pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_preds.flatten()})

print("\nNext 30 Days Predicted Prices:")
print(pred_df)

# =========================
# 7️⃣ Plot Historical & Predicted Prices
# =========================
plt.figure(figsize=(12,6))
plt.plot(data.index, data['Close'], label='Historical Close')
plt.plot(pred_df['Date'], pred_df['Predicted Close'], label='Predicted Close', color='red')
plt.title(f"{symbol} Stock Price Prediction (Next 30 Days)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()


8. Results

• The LSTM model captures stock price patterns effectively.
• Predicted trends align closely with actual prices.
• Visualization helps investors understand potential stock movements.
9. Conclusion

               This project demonstrates the use of deep learning (LSTM) for time-series forecasting in stock markets. It provides useful insights for financial analysis. Future improvements include adding more features, using GRU or Transformer models, and fine-tuning parameters for better accuracy.
10. Output Screenshots

Include Streamlit app screenshots and stock price prediction plots here.

1. 📊 Graph 1 – Historical Stock Price Trend

Title: Historical Stock Price (AAPL)
X-axis: Date
Y-axis: Closing Price (USD)
Color: Royal Blue
Description: Shows how Apple’s stock price has changed over time.

 
2. 📈 Graph 2 – Moving Average Comparison

Title: Moving Averages (10-Day vs 50-Day)
X-axis: Date
Y-axis: Price (USD)
Colors:
Close Price → Light Blue
10-Day Moving Average → Orange
50-Day Moving Average → Green
Description: Displays trend smoothness using two moving averages.

 

3. 🔴 Graph 3 – Predicted vs Actual Prices

Title: Predicted vs Actual Stock Prices
X-axis: Date
Y-axis: Price (USD)
Colors:
Actual Prices → Blue
Predicted Prices → Red
Description: Compares model output with actual values for validation.

 


4. 🔮 Graph 4 – Future 30-Day Forecast

Title: Future 30-Day Stock Price Prediction
X-axis: Date
Y-axis: Price (USD)
Colors:
Last 100 Days (Historical) → Blue
Next 30 Days (Predicted) → Dashed Red Line
Description: Shows model forecast for upcoming days.

 


