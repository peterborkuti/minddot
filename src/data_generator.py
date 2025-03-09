import numpy as np

class DataGenerator:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def generate_image(self):
        image = np.zeros((self.height, self.width), dtype=np.float32)
        y = np.random.randint(0, self.height)
        x = np.random.randint(0, self.width)
        image[y, x] = 1.0  # Set the white pixel
        return image, (y, x)  # Return the image and the coordinates of the white pixel