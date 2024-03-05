from torch import Tensor, nn
import torch
from torch.nn import functional as F
import numpy as np

class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k):
        super(LinearLayer, self).__init__()
        self.vit_base = True
        if k is not None:
            self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])
        else:
            self.vit_base = False
            self.fc = nn.ModuleList([nn.Linear(dim_in * 4 * 2 ** i, dim_out) for i in range(4)])

    def forward(self, tokens):
        for i in range(len(tokens)):
            if self.vit_base:
                tokens[i] = self.fc[i](tokens[i][:, 1:, :])
            else:
                tokens[i] = self.fc[i](tokens[i])
        return tokens


# class LinearLayer(nn.Module):
#     def __init__(self, dim, k):
#         super(LinearLayer, self).__init__()
#         self.fc = nn.ModuleList([nn.Linear(dim, dim) for i in range(k)])
#         self.fc1 = nn.ModuleList([nn.Linear(dim, dim) for i in range(k)])
#
#     def forward(self, tokens, layers):
#         # [1, 197, 512]
#         for i in range(len(layers)):
#             tokens[layers[i]] = self.fc[i](tokens[layers[i]])
#             tokens[layers[i]] = self.fc1[i](tokens[layers[i]])
#         return tokens
