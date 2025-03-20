import cv2
import numpy as np
import torch
import os
import sys
from model import PixelDetectorModel
from predict import predict
from app_config import AppConfig
# Constants
IMAGE_SIZE = AppConfig.image_size
DISPLAY_SCALE = 800//IMAGE_SIZE  # Scale factor for display (32x32 is too small to see)
WINDOW_NAME = "White Pixel Detector Demo"

# Initialize variables
image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
scaled_image = np.zeros((IMAGE_SIZE * DISPLAY_SCALE, IMAGE_SIZE * DISPLAY_SCALE), dtype=np.uint8)

# Load the trained model
def load_model():
    model = PixelDetectorModel()
    model_path = os.path.join(
        os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))), 
                             AppConfig.model_dir, AppConfig.latest_model_name)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Model not found at {model_path}. Using untrained model.")
    
    return model

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    global image, scaled_image
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Convert display coordinates to image coordinates
        img_x = int(x / DISPLAY_SCALE)
        img_y = int(y / DISPLAY_SCALE)
        
        # Ensure coordinates are within bounds
        img_x = max(0, min(img_x, IMAGE_SIZE - 1))
        img_y = max(0, min(img_y, IMAGE_SIZE - 1))
        
        # Reset the image to all black
        image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        
        # Set the clicked pixel to white
        image[img_y, img_x] = 1.0
        
        # Scale the image for display
        scaled_image = cv2.resize(image * 255, 
                                 (IMAGE_SIZE * DISPLAY_SCALE, IMAGE_SIZE * DISPLAY_SCALE), 
                                 interpolation=cv2.INTER_NEAREST)
        
        # Use the model to predict the white pixel location
        prediction = predict(model, image)
        
        if prediction is not None:
            pred_y, pred_x = prediction
            print(f"Original white pixel at: ({img_y}, {img_x})")
            print(f"Predicted white pixel at: ({pred_y:.2f}, {pred_x:.2f})")
            
            # Draw prediction on scaled image as a green circle
            pred_display_x = int(pred_x * DISPLAY_SCALE + DISPLAY_SCALE / 2)
            pred_display_y = int(pred_y * DISPLAY_SCALE + DISPLAY_SCALE / 2)
            
            # Convert to BGR to show colored marker
            scaled_image_color = cv2.cvtColor(scaled_image, cv2.COLOR_GRAY2BGR)
            
            # Draw circle at original position in blue
            orig_display_x = int(img_x * DISPLAY_SCALE + DISPLAY_SCALE / 2)
            orig_display_y = int(img_y * DISPLAY_SCALE + DISPLAY_SCALE / 2)
            cv2.circle(scaled_image_color, (orig_display_x, orig_display_y), 
                     DISPLAY_SCALE // 2, (0, 0, 255), 2)  # Blue circle
            
            # Draw prediction in green
            cv2.circle(scaled_image_color, (pred_display_x, pred_display_y), 
                     DISPLAY_SCALE // 4, (0, 255, 0), 2)  # Green circle
                     
            scaled_image = scaled_image_color
        else:
            print("No white pixel predicted")

# Main function
def main():
    global model, scaled_image
    
    # Load the model
    model = load_model()
    
    # Create window and set mouse callback
    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)
    
    # Scale the initial image for display
    scaled_image = cv2.resize(image * 255, 
                             (IMAGE_SIZE * DISPLAY_SCALE, IMAGE_SIZE * DISPLAY_SCALE), 
                             interpolation=cv2.INTER_NEAREST)
    
    print("Click on the image to place a white pixel. Press ESC to exit.")
    
    while True:
        # Display the scaled image
        cv2.imshow(WINDOW_NAME, scaled_image)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # ESC key to exit
        if key == 27:
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()