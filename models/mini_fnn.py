import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F


def _fnn(input_shape, num_classes, pretrained=False, model_dir="models", depth=2, width=100, nonlinearity=nn.ReLU(), batchnorm=False):
    size = np.prod(input_shape)
    # Linear feature extractor
    modules = [nn.Flatten()]
    modules.append(nn.Linear(size, width))
    if batchnorm:
        modules.append(nn.BatchNorm1d(width))
    modules.append(nonlinearity)
    for i in range(depth - 2):
        modules.append(nn.Linear(width, width))
        if batchnorm:
            modules.append(nn.BatchNorm1d(width))
        modules.append(nonlinearity)

    # Linear classifier
    modules.append(nn.Linear(width, num_classes))
    model = nn.Sequential(*modules)

    # Pretrained model
    if pretrained:
        pretrained_path = f"{model_dir}/fc-bn_mnist{num_classes}.pt"
        pretrained_dict = torch.load(pretrained_path)
        pretrained_dict = pretrained_dict["model_state_dict"] # necessary because of our ckpt format
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    return model

def fnn2(input_shape, num_classes, pretrained=False, model_dir="pretrained_models"):
    return _fnn(input_shape, num_classes, pretrained=pretrained, model_dir=model_dir, depth=2, batchnorm=False)

def fnn2_bn(input_shape, num_classes, pretrained=False, model_dir="pretrained_models"):
    return _fnn(input_shape, num_classes, pretrained=pretrained, model_dir=model_dir, depth=2, batchnorm=True)

def fnn4(input_shape, num_classes, pretrained=False, model_dir="pretrained_models"):
    return _fnn(input_shape, num_classes, pretrained=pretrained, model_dir=model_dir, depth=4, batchnorm=False)

def fnn4_bn(input_shape, num_classes, pretrained=False, model_dir="pretrained_models"):
    return _fnn(input_shape, num_classes, pretrained=pretrained, model_dir=model_dir, depth=4, batchnorm=True)

def fnn6(input_shape, num_classes, pretrained=False, model_dir="pretrained_models"):
    return _fnn(input_shape, num_classes, pretrained=pretrained, model_dir=model_dir, depth=6, batchnorm=False)

def fnn6_bn(input_shape, num_classes, pretrained=False, model_dir="pretrained_models"):
    return _fnn(input_shape, num_classes, pretrained=pretrained, model_dir=model_dir, depth=6, batchnorm=True)
