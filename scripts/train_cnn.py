# solar_forecasting_project/scripts/train_cnn.py
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from cnn_model import SolarCNN
import numpy as np

class GSIDataset(Dataset):
    def __init__(self, image_dir, gsi_file):
        self.image_dir = image_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.gsi_values = np.loadtxt(gsi_file, delimiter=',')[:, 1]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.image_dir, self.image_files[idx]))
        img_tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0  # normalize to [0, 1]
        gsi = torch.tensor(self.gsi_values[idx]).float()
        return img_tensor, gsi

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SolarCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    dataset = GSIDataset(
        image_dir='data/processed/2017_12_18',
        gsi_file='GIRASOL_DATASET/2017_12_18/pyranometer/2017_12_18.csv'
    )
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for epoch in range(5):
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
        torch.save(model.state_dict(), 'models/cnn_model.pth')

if __name__ == '__main__':
    train()
