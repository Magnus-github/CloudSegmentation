import torch
import torch.onnx as onnx

import sys
sys.path.append('.')
from utils import str_to_class


def convert_model_to_onnx(model_path: str, model_name: str) -> None:
    # Load the PyTorch model
    model = str_to_class(f"scripts.model:{model_name}")(in_channels=4, num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    dummy_input = torch.randn(1, 4, 224, 224)

    # Export the model to ONNX format
    onnx_path = model_path.replace('.pth', '.onnx')
    onnx.export(model,
                dummy_input,
                onnx_path, 
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input' : {0 : 'batch_size'},
                                'output' : {0 : 'batch_size'}})

    print(f"Model converted to ONNX and saved at: {onnx_path}")


if __name__ == '__main__':
    model_path = 'model/DeepLabv3_best.pth'
    convert_model_to_onnx(model_path, model_name='CloudSegmentationModel')
