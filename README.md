## Perseo: Spanish Optical Character Recognition (OCR)

Perseo is a vision transformer based OCR for the Spanish language.

The architecture is based on [TrOCR](https://arxiv.org/abs/2109.10282). It is trained on the Spanish Wikipedia dataset, using [trdg](https://github.com/Belval/TextRecognitionDataGenerator) to generate the images of the sentences. The model's encoder is initialized with the small version of the encoder described in the TrOCR paper, while the decoder in initialized with the RoBERTa Spanish model available in [Hugging Face](https://huggingface.co/PlanTL-GOB-ES/roberta-base-bne).

#### Status

Version 0.0 is trained using machine typed characters to evaluate its performance. In future versions handwritten characters will be used.

