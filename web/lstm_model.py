# solar_forecasting_project/scripts/lstm_model.py
import torch
import torch.nn as nn

class SolarLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, output_size=1):
        super(SolarLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        output = self.fc(lstm_out[:, -1, :])
        return output