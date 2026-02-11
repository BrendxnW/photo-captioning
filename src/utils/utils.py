import torch
import re
from collections import Counter
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from src.dataset.flickr8k_dataset import Flickr8kDataset

def get_dataloaders(batch_size=64, num_workers=2):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    training_data = Flickr8kDataset(
        csv_file="data/Train/train.csv",
        root_dir="data/Images",
        transform=transform
    )

    test_data = Flickr8kDataset(
        csv_file="data/Test/test.csv",
        root_dir="data/Images",
        transform=transform
    )

    train_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=caption_collate_fn
    )

    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    return train_loader, test_loader

def caption_collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images)

    captions = pad_sequence(
        captions,
        batch_first=True,
        padding_value=0
    )

    return images, captions

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

class Vocabulary:
    def __init__(self, tokens, threshold=2):
        self.threshold = threshold

        self.word2idx = {"<PAD>":0, "UNK":1, "<SOS>":2, "<EOS>":3}
        self.idx2word = {i:w for w, i in self.padding.items()}
        self.idx = 4

        self.freq = Counter(" ".join(tokens).split())

        for word, count in self.freq.items():
            if count >= self.threshold:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def token_to_id(self, token):
        return self.word2idx.get(token, self.word2idx["<UNK>"])

    def id_to_token(self, idx):
        return self.idx2word.get(idx, "<UNK>")