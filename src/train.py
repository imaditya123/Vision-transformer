

import torch.optim as optim
import torch.nn as nn
import config
from dataset import getData
from model import ViT
from utils import train

def training():
    model = ViT(img_size=config.IMG_SIZE,
                patch_size=config.PATCH_SIZE, 
                hidden_dim=config.HIDDEN_DIM, 
                filter_size=config.FILTER_SIZE, 
                num_heads=config.NUM_HEADS, 
                n_layers=config.N_LAYERS, 
                dropout_rate=config.DROPOUT_RATE, 
                num_classes=config.NUM_CLASSES).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()  # Loss function
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)  # Optimizer

    train_loader,test_loader,_=getData(config.IMG_SIZE)
    train(model,train_loader=train_loader,test_loader=test_loader,criterion=criterion,device=config.DEVICE,optimizer=optimizer)


if __name__=="__main__":
    training()