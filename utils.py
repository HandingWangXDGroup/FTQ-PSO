import torch.nn as nn
import torch
from torchvision import models

class Normalize(torch.nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def backward(self):
        return "mean={}, std={}".format(self.mean, self.std)


def normalize_fn(tensor, mean, std):

    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

def get_model(model_name, device):
    model = eval(f"models.{model_name}(pretrained=True)")
    normalize = Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
    model = torch.nn.Sequential(normalize, model)
    model = model.to(device)
    model.eval()
    return model

