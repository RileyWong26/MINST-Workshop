import torch
import torch.nn as nn
import torch.functional as f
from model import CNN
def train_model(model, device, data_loader, loss_func, optimizer, num_epochs):
    train_loss, train_acc = [], []
    for epoch in range(num_epochs):
        runningLoss = 0.0
        correct = 0
        total = 0