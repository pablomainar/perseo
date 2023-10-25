from dataset import OCRDataset
from trainer import Trainer
from transformers import VisionEncoderDecoderModel, BeitImageProcessor, RobertaTokenizer
from PIL import Image
import requests
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torch.utils.data import DataLoader
import os
import time


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = RobertaTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained('microsoft/beit-base-patch16-224', 'PlanTL-GOB-ES/roberta-base-bne')
    model.to(device)
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    train_dataloader = DataLoader(OCRDataset(tokenizer=tokenizer, device=device, test=False), batch_size=1, shuffle=True)
    test_dataloader = DataLoader(OCRDataset(tokenizer=tokenizer, device=device, test=True), batch_size=1, shuffle=True)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    trainer = Trainer(model=model, optimizer=optimizer, device=device, nb_epochs=10)
    trainer.train(train_dataloader=train_dataloader, test_dataloader=test_dataloader)
    

if __name__ == "__main__":
    main()