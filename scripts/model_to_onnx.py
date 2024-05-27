import torch

import torch.onnx as onnx

def convert_model_to_onnx(model_path: str) -> None:
    # Load the PyTorch model
    model = torch.load(model_path)

    # Set the model to evaluation mode
    model.eval()

    # Create dummy input tensor
    dummy_input = torch.randn(1, 4, 224, 224)  # Adjust the shape according to your model's input

    # Export the model to ONNX format
    onnx_path = model_path.replace('.pth', '.onnx')
    onnx.export(model, dummy_input, onnx_path)

    print(f"Model converted to ONNX and saved at: {onnx_path}")

# Usage example
model_path = '/path/to/your/model.pth'
convert_model_to_onnx(model_path)