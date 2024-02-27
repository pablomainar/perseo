from dataset import OCRDataset
from trainer import Trainer
from transformers import VisionEncoderDecoderModel, RobertaTokenizer
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import DataLoader

characters_mode = "handwritten"  # handwritten or typed

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = RobertaTokenizer.from_pretrained('PlanTL-GOB-ES/roberta-base-bne')
model_trocr = VisionEncoderDecoderModel.from_pretrained(
    pretrained_model_name_or_path="microsoft/trocr-small-stage1")
encoder = model_trocr.encoder
encoder.save_pretrained("pretrained_encoder")
encoder_config = encoder.config
encoder_config.image_size = (64, 2304)
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
    encoder_pretrained_model_name_or_path="pretrained_encoder",
    decoder_pretrained_model_name_or_path="PlanTL-GOB-ES/roberta-base-bne'",
    encoder_config=encoder_config)

model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.eos_token_id = tokenizer.sep_token_id
model.config.max_length = 64
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=["query", "value"],
    lora_dropout=0.1,
    bias="none",
    modules_to_save=["classifier"],
)
model = get_peft_model(model, config)
model.to(device)
train_dataset = OCRDataset(
    characters_mode=characters_mode,
    tokenizer=tokenizer,
    device=device,
    test=False)
test_dataset = OCRDataset(
    characters_mode=characters_mode,
    tokenizer=tokenizer,
    device=device,
    test=True)
train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=True)
optimizer = torch.optim.AdamW(params=model.parameters(), lr=5e-5)
trainer = Trainer(
    characters_mode=characters_mode,
    model=model, optimizer=optimizer,
    device=device,
    nb_epochs=1000)
trainer.train(
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader)
