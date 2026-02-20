from torch.utils.data import Dataset
from PIL import Image
import torch
import pandas as pd
import os
from pathlib import Path
from src.utils.text import tokenize, Vocabulary


class FlickrDataset(Dataset):
    """
    Custom photo captioning dataset.
    
    Flickr8kDataset is a photo caption dataset that consists photos and matching images.
    
    Args:
        csv_file (str): Path to the CSV file that consists the image filenames and captions.
        root_dir (str): The root directory that consists all the photos
        vocab (str): The string
        transform (callable, optional): Optional transform to be applied to each image
    """

    def __init__(self, csv_file, root_dir, vocab, transform=None):
        """
        Initialize the custom dataset
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        """
        Returns the number of samples in dataset
        
        Returns:
            int: Numbers of samples in the dataset
        """
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves the image and caption at the given index
        
        Args:
            idx (int): Index of the sample to retrieve

        Returns:
            tuple:
                - Image (tensor): the transformed image
                - Captions (str): the corresponding caption
        """
        img_file = str(self.df.loc[idx, "image"]).strip()
        caption = self.df.loc[idx, "caption"]

        # In case CSV has "Images/xxx.jpg" or "flickr8k/Images/xxx.jpg", keep only filename
        img_file = os.path.basename(img_file)

        img_path = Path(self.root_dir) / img_file

        if not img_path.exists():
            raise FileNotFoundError(
                f"Missing image file.\n"
                f"root_dir: {self.root_dir}\n"
                f"img_file: {repr(img_file)}\n"
                f"full_path: {img_path}\n"
                f"idx: {idx}"
            )

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        tokens = tokenize(str(caption))
        caption_ids = [self.vocab.word2idx["<SOS>"]]
        caption_ids += [self.vocab.token_to_id(tok) for tok in tokens]
        caption_ids += [self.vocab.word2idx["<EOS>"]]

        caption_tensor = torch.tensor(caption_ids, dtype=torch.long)
        return image, caption_tensor