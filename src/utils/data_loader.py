import torch
import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
from src.dataset.flickr_dataset import FlickrDataset
from src.utils.text import build_vocab_from_csv
from src.utils.text import save_vocab, load_vocab


if os.path.exists("vocab.pkl"):
    vocab = load_vocab("vocab.pkl")
else:
    train8 = pd.read_csv("data/flickr8k/Train/train.csv")
    train30 = pd.read_csv("data/flickr30k/Train/train.csv")
    train_all = pd.concat([train8, train30], ignore_index=True)
    train_all.to_csv("data/train_combined.csv", index=False)

    vocab = build_vocab_from_csv("data/train_combined.csv", threshold=2)
    save_vocab(vocab, "vocab.pkl")

def caption_collate_fn(batch):
    """
    Custom collate function for batching image-caption pairs.

    This function stacks image temsors into a single batch tensor and pads
    the variable-length caption temsors so that all captions in the batch 
    have the same sequence length.
    
    Args:
        batch (list of tuples): A list of image, caption pairs
            - image (Tensor): Image tensor
            - caption (Tensor): 1d tensor that consists tokenized caption

    Returns:
        tuple:
            - images: Batched image tensor
            - captions: Padded caption tensor
    """
    images, captions = zip(*batch)
    images = torch.stack(images)

    captions = pad_sequence(
        captions,
        batch_first=True,
        padding_value=0
    )

    return images, captions


def get_dataloaders(batch_size=64, num_workers=2):
    """
    Creates DataLoaders for the Flickr8k training and test datasets.

    This function defines an image preprocessing pipeline, initializes
    the Flickr8kDataset objects from their CSV files, and wraps them in
    PyTorch DataLoaders for batching and iteration during training/evaluation.

    Args:
        batch_size (int): Number of samples per batch. Defaults to 64.
        num_workers (int): Number of subprocesses used for data loading.
        threshold (int):
    """

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    training_data_8k = FlickrDataset(
        csv_file="data/flickr8k/Train/train.csv",
        root_dir="data/flickr8k/Images/",
        vocab=vocab,
        transform=transform
    )

    test_data_8k = FlickrDataset(
        csv_file="data/flickr8k/Test/test.csv",
        root_dir="data/flickr8k/Images/",
        vocab=vocab,
        transform=transform
    )

    val_data_8k = FlickrDataset(
        csv_file="data/flickr8k/Validate/validate.csv",
        root_dir="data/flickr8k/Images/",
        vocab=vocab,
        transform=transform
    )

    training_data_30k = FlickrDataset(
        csv_file="data/flickr30k/Train/train.csv",
        root_dir="data/flickr30k/Images/",
        vocab=vocab,
        transform=transform
    )

    test_data_30k = FlickrDataset(
        csv_file="data/flickr30k/Test/test.csv",
        root_dir="data/flickr30k/Images/",
        vocab=vocab,
        transform=transform
    )

    val_data_30k = FlickrDataset(
        csv_file="data/flickr30k/Validate/validate.csv",
        root_dir="data/flickr30k/Images/",
        vocab=vocab,
        transform=transform
    )

    train_ds = ConcatDataset([training_data_8k, training_data_30k])
    val_ds = ConcatDataset([val_data_8k, val_data_30k])
    test_ds = ConcatDataset([test_data_8k, test_data_30k])

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=caption_collate_fn
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=caption_collate_fn
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=caption_collate_fn
    )

    return train_loader, test_loader, val_loader, vocab