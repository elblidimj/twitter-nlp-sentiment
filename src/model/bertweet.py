from transformers import (
    AutoModelForSequenceClassification,
)
def build_model(device):
    model = AutoModelForSequenceClassification.from_pretrained(
        'vinai/bertweet-base',
        num_labels=2
    )

    model.to(device)
    return model