import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from src.dataset.flickr8k_dataset import Flickr8kDataset
from src.utils.text import Vocabulary, tokenize, build_vocab_from_csv


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


def get_dataloaders(batch_size=64, num_workers=2, threshold=2):
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

    vocab = build_vocab_from_csv("data/Train/train.csv", threshold=threshold)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    training_data = Flickr8kDataset(
        csv_file="data/Train/train.csv",
        root_dir="data/Images",
        vocab=vocab,
        transform=transform
    )

    test_data = Flickr8kDataset(
        csv_file="data/Test/test.csv",
        root_dir="data/Images",
        vocab=vocab,
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

    return train_loader, test_loader, vocab