
import torch


class AppConfig:
    DEVICE=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    image_size: int = 64
    batch_size: int = 32
    num_epochs: int = 20
    samples_multiplier=5
    learning_rate: float = 0.001
    model_dir: str = 'models'
    model_name: str = 'pixel_detector_model'
    latest_model_name: str = 'latest-' + model_name + '-' + str(image_size) + '.pth'
    best_model_name: str = 'best-' + model_name + '-' + str(image_size) + '.pth'