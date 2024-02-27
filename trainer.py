import torch
from tqdm import tqdm
import os


class Trainer():
    def __init__(self, characters_mode, model, optimizer, device, nb_epochs):
        self.characters_mode = characters_mode
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.nb_epochs = nb_epochs
        self.train_losses = []
        self.test_losses = []
        if characters_mode == "handwritten":
            self.path_models = "models_handwritten/"
        elif characters_mode == "typed":
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
            loss = self.model(pixel_values=X,
                              labels=y,
                              decoder_attention_mask=mask).loss
            loss.sum().backward()
            self.optimizer.step()
            running_loss += loss.sum().item()

            if batch % 100 == 0 and batch != 0:
                print(f"Batch {batch}: Loss ", running_loss / 100)
                epoch_loss += running_loss
                running_loss = 0
        epoch_loss += running_loss
        self.train_losses.append(epoch_loss / batch)
        print("Training: Loss ", epoch_loss / batch)

    def test_loop(self, dataloader):
        self.model.eval()
        running_loss = 0
        with torch.no_grad():
            for batch, (X, y, mask) in enumerate(dataloader):
                X = X.to(self.device)
                y = y.to(self.device)
                mask = mask.to(self.device)
                loss = self.model(pixel_values=X,
                                  labels=y,
                                  decoder_attention_mask=mask).loss
                running_loss += loss.sum().item()
        self.test_losses.append(running_loss / batch)
        print("Testing: Loss ", running_loss / batch)

    def save_model(self, epoch):
        torch.save(obj=self.model.state_dict(),
                   f=self.path_models + f"model_{epoch}.pt")
        torch.save(obj=self.optimizer.state_dict(),
                   f=self.path_models + f"optimizer_{epoch}.pt")
        torch.save(obj=self.train_losses,
                   f=self.path_models + f"train_losses_{epoch}.pt")
        torch.save(obj=self.test_losses,
                   f=self.path_models + f"test_losses_{epoch}.pt")
        print(f"Model saved at epoch {epoch}")
