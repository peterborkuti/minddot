import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from app_config import AppConfig
from data_generator import DataGenerator
from model import PixelDetectorModel

class PixelDataset(Dataset):
    def __init__(self, generator, num_samples):
        self.generator = generator
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image, coordinates = self.generator.generate_image(idx)
        return torch.tensor(image, dtype=torch.float32).unsqueeze(0), torch.tensor(coordinates, dtype=torch.float32)

def train_model(num_epochs=AppConfig.num_epochs, batch_size=AppConfig.batch_size, learning_rate=AppConfig.learning_rate):
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), AppConfig.model_dir)
    os.makedirs(save_dir, exist_ok=True)

    generator = DataGenerator(AppConfig.image_size,AppConfig.image_size)
    num_samples = AppConfig.samples_multiplier*len(generator)
    dataset = PixelDataset(generator, num_samples)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_batches = len(dataloader)

    model = PixelDetectorModel()
    model.to(AppConfig.DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting training with {num_samples} samples over {num_epochs} epochs")
    print(f"Image size: {AppConfig.image_size}x{AppConfig.image_size}, Batch size: {batch_size}")
    
    # Track best model
    best_loss = float('inf')
    best_model_state = None

    # Create progress bar for epochs
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs")
    for epoch in epoch_pbar:
        model.train()
        running_loss = 0.0
        
        # Create progress bar for batches
        batch_pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for i, (images, coordinates) in enumerate(batch_pbar):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, coordinates)
            loss.backward()
            optimizer.step()
            
            # Update statistics
            running_loss += loss.item()
            current_loss = running_loss / (i + 1)
            
            # Update batch progress bar
            batch_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{current_loss:.4f}'
            })
        
        # Calculate average loss for the epoch
        epoch_loss = running_loss / total_batches
        best = False

        # best model
        if epoch_loss < best_loss:
            best = True
            best_loss = epoch_loss
            best_model_state = model.state_dict().copy()
            # Save a copy as latest model for easy reference
            best_model_path = os.path.join(save_dir, AppConfig.best_model_name)
            torch.save(best_model_state, best_model_path)  # Use the best model for latest
            print(f"Best model saved to {best_model_path}")
        
        epoch_pbar.set_postfix({
                'loss': f'{epoch_loss:.4f}',
                'best': best
        })
    
    print(f"Training completed. Best loss: {best_loss:.4f}")
    
    # Save the final model
    model_path = os.path.join(save_dir, AppConfig.latest_model_name)
    torch.save(model.state_dict(), model_path)
    
    print(f"Latest model saved to {model_path}")


if __name__ == "__main__":
    train_model()