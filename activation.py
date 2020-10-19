"""
MILQT - "Multiple interaction learning with question-type prior knowledge for constraining answer search space in visual question answering."
Do, Tuong, Binh X. Nguyen, Huy Tran, Erman Tjiputra, Quang D. Tran, and Thanh-Toan Do.
Our arxiv link: https://arxiv.org/abs/2009.11118

This code is written by Tuong Do.
"""


import torch
import torch.nn as nn

"""Define activation for VQA"""

"""Swish from Searching for Activation Functions
 https://arxiv.org/abs/1710.05941"""
class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return torch.mul(x, torch.sigmoid(x))