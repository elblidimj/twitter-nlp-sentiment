import re
import emoji
import torch
from torch.utils.data import Dataset, DataLoader
class BERTweetNormalizer:
    """Optimized normalizer using compiled regex"""

    def __init__(self):
        self.url_pattern = re.compile(
            r'<url>'
        )
        self.mention_pattern = re.compile(r'<user>')
        self.whitespace_pattern = re.compile(r'\s+')

    def normalize(self, text):
        if not isinstance(text, str) or not text:
            return ""
        text = emoji.demojize(text, delimiters=("", ""))
        text = self.url_pattern.sub('HTTPURL', text)
        text = self.mention_pattern.sub('@USER', text)
        text = self.whitespace_pattern.sub(' ', text).strip()
        return text

    def normalize_batch(self, texts):
        return [self.normalize(t) for t in texts]
    
class OptimizedTweetDataset(Dataset):
    """Memory-efficient dataset"""

    def __init__(self, texts, labels, tokenizer, max_length, cache_encodings=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_encodings = cache_encodings
        self._cache = {}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.cache_encodings and idx in self._cache:
            return self._cache[idx]

        encoding = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

        if self.cache_encodings:
            self._cache[idx] = item

        return item