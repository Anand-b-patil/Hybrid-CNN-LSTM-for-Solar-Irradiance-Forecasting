import torch
import torch.nn as nn

class SolarCNN(nn.Module):
    def __init__(self):
        super(SolarCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # Input: 3x240x320
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x120x160

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x60x80

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)   # 64x30x40
        )

        # Automatically calculate the size of the flattened tensor
        self._to_linear = self._get_linear_input_size((3, 240, 320))
        
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._to_linear, 128),  # Adjusted size here
            nn.ReLU(),
            nn.Linear(128, 1)  # Output: GSI (scalar)
        )

    def _get_linear_input_size(self, shape):
        """ Helper function to calculate the flattened size after convolutions """
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)  # A dummy input tensor
            dummy_output = self.features(dummy_input)  # Pass it through the convolution layers
            return dummy_output.numel()  # Flattened size

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x
