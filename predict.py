import yfinance as yf
import numpy as np
import torch
import torch.nn as nn

from sklearn.preprocessing import MinMaxScaler

# Download stock data
df = yf.download("AAPL", start="2015-01-01", end="2025-01-01")

data = df[['Close']]

# Scale data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)

# Prepare sequences
X = []
y = []

for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i, 0])
    y.append(scaled_data[i, 0])

X = np.array(X)
y = np.array(y)

X_train = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y, dtype=torch.float32).view(-1,1)

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

        self.fc = nn.Linear(50,1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# Create model
model = StockLSTM()

# Train model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 5

for epoch in range(epochs):

    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Predict next day
last_60_days = scaled_data[-60:]

X_test = np.array([last_60_days])

X_test = torch.tensor(X_test, dtype=torch.float32)

prediction = model(X_test).detach().numpy()

# Convert back to real price
predicted_price = scaler.inverse_transform(prediction)

print("Predicted Next Day Price:")
print(predicted_price[0][0])