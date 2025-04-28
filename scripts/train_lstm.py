# solar_forecasting_project/scripts/train_lstm.py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from lstm_model import SolarLSTM


class GSITimeSeriesDataset(Dataset):
    def __init__(self, gsi_values, sequence_length=10):
        self.sequence_length = sequence_length
        self.data = gsi_values.astype(np.float32)

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.sequence_length]
        y = self.data[idx + self.sequence_length]
        return torch.tensor(x).unsqueeze(-1), torch.tensor(y)


def train():
    # === Hyperparameters ===
    sequence_length = 10
    batch_size = 32
    learning_rate = 1e-3
    num_epochs = 10
    hidden_size = 128

    # === Load Data ===
    csv_path = 'GIRASOL_DATASET/2017_12_18/pyranometer/2017_12_18.csv'
    df = pd.read_csv(csv_path)
    gsi_values = df.iloc[:, 1].values  # Assuming second column is GSI

    dataset = GSITimeSeriesDataset(gsi_values, sequence_length=sequence_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # === Model, Optimizer, Loss ===
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SolarLSTM(input_size=1, hidden_size=hidden_size, output_size=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # === Training Loop ===
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output.squeeze(), y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(loader):.4f}")

    # === Save Model ===
    torch.save(model.state_dict(), 'models/lstm_model.pth')


if __name__ == '__main__':
    train()
