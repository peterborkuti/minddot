import torch

from app_config import AppConfig

class PixelDetectorModel(torch.nn.Module):
    def __init__(self):
        super(PixelDetectorModel, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, padding=1)
        # After two max pooling layers (each reducing dimensions by half),
        # your feature maps would be image_size/4 × image_size/4. With 32 channels
        self.fc1 = torch.nn.Linear(32 * (AppConfig.image_size // 4) * (AppConfig.image_size // 4), 128)  # Assuming input size is 32x32
        self.fc2 = torch.nn.Linear(128, 2)  # Output coordinates (x, y)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = torch.nn.functional.relu(self.conv2(x))
        x = torch.nn.functional.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc2(x)  # Output layer
        return x