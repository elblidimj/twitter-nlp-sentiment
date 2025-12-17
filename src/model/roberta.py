import torch
from transformers import RobertaForSequenceClassification

class RobertaClassifier(torch.nn.Module):
    def __init__(self, model_name="roberta-base"):
        super().__init__()
        self.model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
