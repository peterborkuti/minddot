# White Pixel Detector

This project implements a neural network-based solution to detect the coordinates of a single white pixel in a black and white image. The images are generated randomly, ensuring that each image contains only one white pixel, represented by the value `1`, while all other pixels are black, represented by the value `0`.

## Project Structure

```
white-pixel-detector
├── src
│   ├── data_generator.py      # Generates random black and white images with one white pixel
│   ├── model.py               # Defines the neural network model for pixel detection
│   ├── train.py               # Handles the training process of the model
│   └── predict.py             # Contains the prediction function to find the white pixel
├── requirements.txt           # Lists the project dependencies
├── .gitignore                 # Specifies files to be ignored by Git
└── README.md                  # Documentation for the project
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd white-pixel-detector
   ```

2. **Install the required dependencies:**
   Create a virtual environment (optional but recommended) and install the dependencies listed in `requirements.txt`:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Generate Images:**
   Use the `DataGenerator` class from `src/data_generator.py` to create random images with a single white pixel.

2. **Train the Model:**
   Run the training script to train the neural network model:
   ```
   python src/train.py
   ```

3. **Make Predictions:**
   After training, use the `predict` function from `src/predict.py` to find the coordinates of the white pixel in a given image.

## File Descriptions

- **data_generator.py:** Contains the `DataGenerator` class that generates random images with a single white pixel.
- **model.py:** Defines the `PixelDetectorModel` class, which is a neural network for detecting the white pixel.
- **train.py:** Manages the training process, including data loading, model training, and evaluation.
- **predict.py:** Implements the prediction logic to identify the coordinates of the white pixel in an input image.

## License

This project is licensed under the MIT License - see the LICENSE file for details.