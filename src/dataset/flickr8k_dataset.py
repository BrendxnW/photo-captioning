from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import os


class Flickr8kDataset(Dataset):
    """
    Custom photo captioning dataset.
    
    Flickr8kDataset is a photo caption dataset that consists photos and matching images.
    
    Args:
        csv_file (str): Path to the CSV file that consists the image filenames and captions.
        root_dir (str): The root directory that consists all the photos
        transform (callable, optional): Optional transform to be applied to each image
    """

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Initialize the custom dataset
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
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
        img_file = self.df.loc[idx, "image"]
        caption = self.df.loc[idx, "caption"]

        img_path = os.path.join(self.root_dir, img_file)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, caption