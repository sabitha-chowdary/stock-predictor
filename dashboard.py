import streamlit as st
import yfinance as yf
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

# ---------------------------
# Streamlit Page Config
# ---------------------------

st.set_page_config(page_title="AI Stock Predictor", layout="wide")

st.title("📈 AI Stock Price Prediction Dashboard")

# ---------------------------
# Sidebar
# ---------------------------

st.sidebar.header("Stock Selection")

ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")

# ---------------------------
# Download Data
# ---------------------------

df = yf.download(ticker, start="2015-01-01", end="2025-01-01")

# ---------------------------
# Moving Averages
# ---------------------------

df['MA50'] = df['Close'].rolling(50).mean()
df['MA200'] = df['Close'].rolling(200).mean()

# ---------------------------
# Display Data
# ---------------------------

st.subheader("Latest Stock Data")
st.write(df.tail())

# ---------------------------
# Scale Data
# ---------------------------

data = df[['Close']]

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# ---------------------------
# Create Sequences
# ---------------------------

X = []
y = []

for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

X_train = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y, dtype=torch.float32).view(-1,1)

# ---------------------------
# LSTM Model
# ---------------------------

class StockLSTM(nn.Module):

    def __init__(self):
        super(StockLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=50,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(50,1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ---------------------------
# Train Model
# ---------------------------

model = StockLSTM()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# ---------------------------
# Predict Next Day
# ---------------------------

last_60_days = scaled_data[-60:]

X_test = np.array([last_60_days])

X_test = torch.tensor(X_test, dtype=torch.float32)

prediction = model(X_test).detach().numpy()

predicted_price = scaler.inverse_transform(prediction)

# ---------------------------
# Prediction Output
# ---------------------------

st.subheader("🔮 AI Predicted Next Day Price")

st.success(f"Predicted Price: ${predicted_price[0][0]:.2f}")

# ---------------------------
# Stock Chart
# ---------------------------

st.subheader("📊 Stock Price Chart")

fig, ax = plt.subplots(figsize=(14,6))

ax.plot(df['Close'], label='Close Price')
ax.plot(df['MA50'], label='50-Day MA')
ax.plot(df['MA200'], label='200-Day MA')

ax.set_title(f"{ticker} Stock Analysis")
ax.set_xlabel("Date")
ax.set_ylabel("Price")

ax.legend()

st.pyplot(fig)

# ---------------------------
# Volume Chart
# ---------------------------

st.subheader("📉 Trading Volume")

fig2, ax2 = plt.subplots(figsize=(14,4))

ax2.plot(df['Volume'])

ax2.set_title("Trading Volume")

st.pyplot(fig2)