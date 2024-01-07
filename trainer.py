import torch
import numpy as np
from tqdm import tqdm
import os
import time

class Trainer():
    def __init__(self, model, optimizer, device, nb_epochs):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.nb_epochs = nb_epochs
        self.train_losses = []
        self.test_losses = []
        self.path_models = "models/"
        if not os.path.exists(self.path_models):
            os.mkdir(self.path_models)

    
    def train(self, train_dataloader, test_dataloader):
        for epoch in range(self.nb_epochs):
            print(f"Epoch {epoch}")
            self.train_loop(train_dataloader)
            self.test_loop(test_dataloader)
            if epoch % 1 == 0:
                self.save_model(epoch)

    def train_loop(self, dataloader):
        self.model.train()
        running_loss = 0
        epoch_loss = 0
        for batch, (X, y, mask) in tqdm(enumerate(dataloader)):
            self.optimizer.zero_grad()
            X = X.to(self.device)
            y = y.to(self.device)
            mask = mask.to(self.device)
            loss = self.model(pixel_values=X, labels=y, decoder_attention_mask=mask).loss
            loss.sum().backward()
            self.optimizer.step()
            running_loss += loss.sum().item()

            if batch % 100 == 0 and batch != 0:
                print(f"Batch {batch}: Loss ", running_loss / 100)
                epoch_loss += running_loss
                running_loss = 0
        epoch_loss += running_loss
        self.train_losses.append(epoch_loss / batch)
        print(f"Training: Loss ", epoch_loss / batch)
        
    def test_loop(self, dataloader):
        self.model.eval()
        running_loss = 0
        with torch.no_grad():
            for batch, (X,y, mask) in enumerate(dataloader):
                X = X.to(self.device)
                y = y.to(self.device)
                mask = mask.to(self.device)
                loss = self.model(pixel_values=X, labels=y, decoder_attention_mask=mask).loss
                running_loss += loss.sum().item()
        self.test_losses.append(running_loss / batch)
        print(f"Testing: Loss ", running_loss / batch)

    def save_model(self, epoch):
        torch.save(self.model.state_dict(), self.path_models + f"model_{epoch}.pt")
        torch.save(self.optimizer.state_dict(), self.path_models + f"optimizer_{epoch}.pt")
        torch.save(self.train_losses, self.path_models + f"train_losses_{epoch}.pt")
        torch.save(self.test_losses, self.path_models + f"test_losses_{epoch}.pt")
        print(f"Model saved at epoch {epoch}")
