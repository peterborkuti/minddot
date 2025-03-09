def predict(model, image):
    import torch
    import numpy as np

    # Ensure the model is in evaluation mode
    model.eval()

    # Convert the image to a tensor and add a batch dimension
    input_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Forward pass through the model
    with torch.no_grad():
        output = model(input_tensor)

    # Get the output as a numpy array
    output_np = np.round(output.squeeze().numpy())

    # Find the coordinates of the white pixel (value 1)
    return output_np



 