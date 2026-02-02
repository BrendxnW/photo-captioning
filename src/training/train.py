import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from src.utils.utils import get_dataloaders



train_loader, test_loader = get_dataloaders(batch_size=64, num_workers=2)

resnet_model = models.resnet18(weights="IMAGENET1K_V1")
feat_extract = nn.Sequential(*list(resnet_model.children())[:-1])

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(resnet_model.parameters(), lr=0.001)
running_loss = 0.0

for i, (image, caption) in enumerate(get_dataloaders.train_loader):
    output = resnet_model(image)
    loss = loss_function(output, caption)

    print(f"Batch {i+1:5d} training loss: {loss.item():.4f}")

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

print("Finished Training")