import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from src.utils.utils import get_dataloaders, caption_collate_fn

class PhotoCaptioner(nn.Module):
    def __init__(self, encoder, vocab_size):
        super().__init__()
        self.encoder = encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.projection = nn.Linear(2048, 512)
        self.decoder = nn.LSTM(input_size=512, hidden_size=512, batch_first=True)
        self.fc_out = nn.Linear(512, vocab_size)

    def forward(self, images, captions_in):
        with torch.no_grad():
            feature = self.encoder(images)
            feature = torch.flatten(1)

        projected_feats = self.projection(feature)
        projected_caption = self.projection(captions_in)

        projected_feats = projected_feats.unsqueeze(1)
        x = torch.cat([projected_feats, projected_caption], dim=1)

        outputs, _ = self.decoder(x)
        stats = self.fc_out(outputs)
        return stats    

def train_one_epoch(model, loader, optimizer, device):
    model.train()
    loss_function = nn.CrossEntropyLoss(ignore_index=0)

    for i, (images, captions) in enumerate(loader):
        images = images.to(device)
        captions = captions.to(device)

        captions_in = captions[:, :-1]
        targets = captions[:, 1:]  

        optimizer.zero_grad()
        outputs = model(images, captions_in)

        outputs = outputs[:, 1:, :]

        loss = loss_function(-1, outputs.size(-1),
                             targets.reshape(-1)
                             )
        
        loss.backward()
        optimizer.step()
        print(f"Batch {i+1:5d} training loss: {loss.item():.4f}")

    print("Finished Training")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    train_loader, test_loader = get_dataloaders(batch_size=64, num_workers=0)

    vocab_size = 5000

    resnet_model = models.resnet50(weights="IMAGENET1K_V1")
    feat_extract = nn.Sequential(*list(resnet_model.children())[:-1])
    
    model = PhotoCaptioner(feat_extract, vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(5):
        train_one_epoch(model, train_loader, optimizer, device)

if __name__ == "__main__":
    main()