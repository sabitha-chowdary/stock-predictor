import yfinance as yf
import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

# Download stock data
df = yf.download("AAPL", start="2015-01-01", end="2025-01-01")

# Keep only Close prices
data = df[['Close']]

# Normalize data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# Create sequences
X = []
y = []

for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i, 0])
    y.append(scaled_data[i, 0])

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Convert to tensors
X_train = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# LSTM Model
class StockLSTM(nn.Module):

    def __init__(self):
        super(StockLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=50,
            num_layers=2,
            batch_first=True
        )

        self.fc = nn.Linear(50, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Create model
model = StockLSTM()

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train model
epochs = 10

for epoch in range(epochs):

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

print("Model training complete!")

# Predictions
model.eval()

with torch.no_grad():
    predicted = model(X_train).numpy()

# Convert back to real prices
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y.reshape(-1,1))

# Plot graph
plt.figure(figsize=(12,6))

plt.plot(real_prices, label="Actual Price")
plt.plot(predicted_prices, label="Predicted Price")

plt.title("Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Price")

plt.legend()

plt.show()