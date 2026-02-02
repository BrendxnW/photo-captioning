from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os


class Flickr8kDataset(Dataset):
    """Photo captioning dataset"""

    def __init__(self, csv_file, root_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_file = self.df.loc[idx, "image"]
        caption = self.df.loc[idx, "caption"]

        img_path = os.path.join(self.root_dir, img_file)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, caption