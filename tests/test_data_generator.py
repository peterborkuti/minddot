import numpy as np
from src.data_generator import DataGenerator

def test_image_size():
    generator = DataGenerator(5, 5)
    image, coordinates = generator.generate_image(0)
    assert image.shape == (5, 5)
    generator = DataGenerator(12, 8)
    image, coordinates = generator.generate_image(0)
    assert image.shape == (12, 8)

def test_coordinates():
    generator = DataGenerator(5, 5)
    image, coordinates = generator.generate_image(0)
    assert coordinates == (0, 0)
    image, coordinates = generator.generate_image(1)
    assert coordinates == (0, 1)
    image, coordinates = generator.generate_image(5)
    assert coordinates == (1, 0)
    image, coordinates = generator.generate_image(24)
    assert coordinates == (4, 4)
    image, coordinates = generator.generate_image(25)
    assert coordinates == (0, 0)
    image, coordinates = generator.generate_image(26)
    assert coordinates == (0, 1)