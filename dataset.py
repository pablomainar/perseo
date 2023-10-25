from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Resize
from transformers import BeitImageProcessor
import torch
import os

class OCRDataset(Dataset):
    def __init__(self, tokenizer, device, test=False):
        self.tokenizer = tokenizer
        self.device = device
        self.image_resizer = Resize((32, 1200), antialias=True)
        if test:
            self.data_path = "data/test/out/"
        else:
            self.data_path = "data/train/out/"
        self.files = os.listdir(self.data_path)
        self.max_length = 0
        for f in self.files:
            f = f.split("_")[0]
            tokenizer_output = self.tokenizer(f)
            if len(tokenizer_output['input_ids']) > self.max_length:
                self.max_length = len(tokenizer_output['input_ids'])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.files[idx])
        #img_path = os.path.join(self.data_path, self.files[0]) # CAREFUL: THIS IS FIXED AND IT SHOULD NOT
        image = read_image(img_path)
        image = image.float() / 255
        pixel_values = self.image_resizer(image)
        pixel_values = pixel_values.to(self.device)
        tokenizer_output = self.tokenizer(self.files[idx].split("_")[0])
        #tokenizer_output = self.tokenizer(self.files[0].split("_")[0]) # CAREFUL: THIS IS FIXED AND IT SHOULD NOT
        labels = tokenizer_output['input_ids']
        pad_length = self.max_length - len(labels) 
        labels = torch.tensor(labels + [-100] * pad_length)
        labels = labels.to(self.device)
        attention_masks = tokenizer_output['attention_mask']
        attention_masks = attention_masks + [0] * pad_length
        attention_masks = torch.tensor(attention_masks)
        return pixel_values, labels, attention_masks