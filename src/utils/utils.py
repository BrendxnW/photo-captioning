import torch
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