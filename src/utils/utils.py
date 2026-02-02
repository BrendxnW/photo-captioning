from torchvision import transforms
from src.dataset.flickr8k_dataset import Flickr8kDataset


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