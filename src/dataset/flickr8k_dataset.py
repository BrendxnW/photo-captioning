from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os


class Flickr8kDataset(Dataset):
    """Photo captioning dataset"""

    def __init__(self, csv_file, root_dir, transfrom=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transfrom = transfrom

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        idx = int(idx)
        
        img_file = self.df.iloc([idx, 0])
        caption = self.df.iloc([idx, 1])

        img_path = os.path.join(self.root_dir, img_file)
        image = Image.open(img_path).conver("RGB")

        sample = {'image': image, 'captions': caption}
        
        if self.transform:
            sample["image"] = self.tansfrom(sample["image"])

        return sample