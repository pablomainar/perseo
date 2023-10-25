import torch

class Trainer():
    def __init__(self, model, optimizer, device, nb_epochs):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.nb_epochs = nb_epochs
    
    def train(self, train_dataloader, test_dataloader):
        for epoch in range(self.nb_epochs):
            print(f"Epoch {epoch}")
            self.train_loop(train_dataloader)
            self.test_loop(test_dataloader)

    def train_loop(self, dataloader):
        self.model.train()
        for batch, (X, y, mask) in enumerate(dataloader):
            self.optimizer.zero_grad()
            X = X.to(self.device)
            y = y.to(self.device)
            loss = self.model(pixel_values=X, labels=y).loss
            loss.backward()
            self.optimizer.step()
            if batch % 100 == 0:
                print(f"Training: Batch {batch} Loss {loss.item()}")
        
    def test_loop(self, dataloader):
        self.model.eval()
        with torch.no_grad():
            for batch, (X,y, mask) in enumerate(dataloader):
                X = X.to(self.device)
                y = y.to(self.device)
                loss = self.model(pixel_values=X, labels=y).loss
                if batch % 100 == 0:
                    print(f"Testing: Batch {batch} Loss {loss.item()}")