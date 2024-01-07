from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Resize
from transformers import BeitImageProcessor
import torch
import os
import pandas as pd
import time

class OCRDataset(Dataset):
    def __init__(self, tokenizer, device, test=False):
        self.tokenizer = tokenizer
        self.device = device
        #self.image_resizer = Resize((32, 1568), antialias=True)
        #self.image_resizer = Resize((224, 224), antialias=True) # beit
        #self.image_resizer = Resize((384, 384), antialias=True) #vit
        self.image_resizer = Resize((64,2304), antialias=True)  
        if test:
            self.data_path = "data_test/data"
            self.labels_path = "data_test/test_tokens.csv"
        else:
            self.data_path = "data_all"
            self.labels_path = "data_all/train_tokens.csv"
        self.labels_df = pd.read_csv(self.labels_path)
        self.max_length = int(self.labels_df["token_length"].max())   
       

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.labels_df.iloc[idx]["path"])
        image = read_image(img_path)
        image = image.float() / 255
        pixel_values = self.image_resizer(image)

        labels = self.labels_df.iloc[idx]["tokens"]
        labels = labels[1:-1]
        labels = labels.split(", ")
        labels = [int(label) for label in labels]   
        pad_length = self.max_length - len(labels) 
        labels = torch.tensor(labels + [-100] * pad_length)

        attention_masks = self.labels_df.iloc[idx]["attention_mask"]
        attention_masks = attention_masks[1:-1]
        attention_masks = attention_masks.split(", ")
        attention_masks = [int(mask) for mask in attention_masks]
        attention_masks = attention_masks + [0] * pad_length
        attention_masks = torch.tensor(attention_masks)
        return pixel_values, labels, attention_masks