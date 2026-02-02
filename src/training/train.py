import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from src.utils.utils import get_dataloaders

device = torch.device("cuda")

def main():
    train_loader, test_loader = get_dataloaders(batch_size=64, num_workers=2)

    resnet_model = models.resnet50(weights="IMAGENET1K_V1")
    feat_extract = nn.Sequential(*list(resnet_model.children())[:-1])
    feat_extract.to(device)

    with torch.no_grad():
        images, captions = next(iter(train_loader))
        images = images.to(device)

        feats = feat_extract(images)
        feats = feats.reshape(images.size(0), -1)

    print("features shape:", feats.shape)
    print("caption example:", captions[0])

if __name__ == "__main__":
    main()