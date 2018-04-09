import torch
from torchvision.models import *
import torch.nn as nn
from torch.autograd import Variable

class EmbeddingLayer(nn.Module):
    def __init__(self, f_dims=128):
        super(EmbeddingLayer, self).__init__()
        self.base_model = resnet34(pretrained=True)
        self.base_model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_in_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(in_features=self.fc_in_features, out_features=f_dims)

    def l2_norm(self,input):
        input_size = input.size()
        buffer = torch.pow(input, 2)

        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)

        _output = torch.div(input, norm.view(-1, 1).expand_as(input))

        output = _output.view(input_size)

        return output

    def forward(self, x):
        x = self.base_model(x)
        x = self.l2_norm(x)
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, f_dims=128):
        super(SiameseNetwork, self).__init__()
        self.f_dims = f_dims
        self.embedding = EmbeddingLayer(self.f_dims)

    def forward(self, x):
        return self.embedding(x)