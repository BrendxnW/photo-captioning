from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from src.dataset.flickr8k_dataset import Flickr8kDataset


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

training_data = Flickr8kDataset(
    csv_file="data/Train/train.csv",
    root="data/Images",
    transform=transform
)

test_data = Flickr8kDataset(
    csv_file="data/Test/test.csv",
    root="data/Images",
    transform=transform
)

import matplotlib.pyplot as plt
import random

idx = random.randint(0, len(training_data) - 1)
img, label = training_data[idx]

plt.imshow(img.permute(1,2,0))
plt.title(training_data.classes[label])
plt.axis("off")
plt.show()