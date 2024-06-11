import torch.onnx
from Network import CNN
import onnx

model = CNN()

torch_input = torch.randn(1, 1, 28, 28)
torch.onnx.export(model, torch_input,"mnist_classifier.onnx",verbose=True)
