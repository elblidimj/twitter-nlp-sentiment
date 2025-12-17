from transformers import RobertaTokenizer
import torch

class RobertaTweetTokenizer:
    def __init__(self, model_name="roberta-base", max_length=128):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def encode(self, tweets):
        return self.tokenizer(
            tweets,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
