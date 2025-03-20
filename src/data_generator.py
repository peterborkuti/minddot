import numpy as np

class DataGenerator:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __len__(self):
        return self.height * self.width

    def generate_image(self, idx):
        image = np.zeros((self.height, self.width), dtype=np.float32)
        index = idx % (self.height * self.width)
        x = index % self.width
        y = index // self.width
        image[y, x] = 1.0  # Set the white pixel
        return image, (y, x)  # Return the image and the coordinates of the white pixel