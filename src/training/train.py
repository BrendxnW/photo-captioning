import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from src.utils.data_loader import get_dataloaders, caption_collate_fn
from src.utils.text import Vocabulary, tokenize

class PhotoCaptioner(nn.Module):
    """
    Image captioning model using a frozen CNN encoder and an LSTM decoder.

    The encoder extracts a feature vector from each image. These image features 
    are projected into the decoder embedding space and then concatenated with the
    embedded caption tokens to predict the next token at each time step.

    Args:
        encoder (nn.Module): CNN feature extractor that outputs image features.
        vocab_size (int): Number of tokens in the vocabulary.
    """
    def __init__(self, encoder, vocab_size):
        """
        Initializes the PhotoCaptioner model..
        """
        super().__init__()
        self.encoder = encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.projection = nn.Linear(2048, 512)          # image feats -> 512
        self.embed = nn.Embedding(vocab_size, 512, padding_idx=0)  # caption ids -> 512

        self.decoder = nn.LSTM(input_size=512, hidden_size=512, batch_first=True)
        self.fc_out = nn.Linear(512, vocab_size)


    def forward(self, images, captions_in):
        """
        Forward pass for caption generation training.

        Args:
            images (Tensor): Batch of images.
            captions_in (Tensor): Input caption token.

        Returns:
            Tensor: Logits over the vocabulary for each timestep.
        """
        with torch.no_grad():
            feature = self.encoder(images)    
            feature = torch.flatten(feature, 1)    

        projected_feats = self.projection(feature) 
        caption_embed = self.embed(captions_in) 

        projected_feats = projected_feats.unsqueeze(1)
        x = torch.cat([projected_feats, caption_embed], dim=1)

        outputs, _ = self.decoder(x)
        logits = self.fc_out(outputs)
        return logits    


def train_one_epoch(model, loader, optimizer, device):
    """
    Trains the model for a single epoch.

    Iterates over the dataloader, performs a forward pass with teacher forcing,
    computes cross-entropy loss, backpropagates, and updates model parameters.

    Args:
        model (nn.Module): The captioning model to train.
        loader (DataLoader): Dataloader providing (images, captions) batches. 
                             Captions are expected to be padded token index tensors.
        optimizer (torch.optim.Optimizer): Optimizer used to update model parameters.
        device (torch.device): Device to run training on (CPU or CUDA).
    """
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

        loss = loss_function(
            outputs.reshape(-1,
            outputs.size(-1)),
            targets.reshape(-1)
            )
        
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print(f"Batch {i+1:5d} training loss: {loss.item():.4f}")

    print("Finished Training")


def evaluate(model, loader, device, pad_idx=0):
    """
    Docstring for evaluate
    
    :param model: Description
    :param loader: Description
    :param device: Description
    """
    model.eval()
    loss_function = nn.CrossEntropyLoss(ignore_index=0)

    total_loss = 0.0
    correct = 0
    total = 0

    for images, captions in loader:
        images = images.to(device)
        captions = captions.to(device).long()

        captions_in = captions[:, :-1]
        targets = captions[:, 1:]

        outputs = model(images, captions_in)
        outputs = outputs[:, 1:, :]

        loss = loss_function(
            outputs.reshape(-1,
            outputs.size(-1)),
            targets.reshape(-1)
        )

        total_loss += loss.item()
        prediction = outputs.argmax(dim=-1)
        mask = targets != pad_idx

        correct += (prediction[mask] == targets[mask]).sum().item()
        total += mask.sum().item()

    avg_loss = total_loss / len(loader)
    token_acc = 100.0 * correct / max(total, 1)
    
    return avg_loss, token_acc

def main():
    """
    Entry point for training the image captioning model.

    Creates the device, builds dataloaders, constructs a ResNet-based encoder,
    initializes the PhotoCaptioner model, and trains for a fixed number of epochs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    train_loader, test_loader, val_loader, vocab = get_dataloaders(batch_size=64, num_workers=0, threshold=2)
    vocab_size = len(vocab.word2idx)
    num_epoch = 1

    resnet_model = models.resnet50(weights="IMAGENET1K_V1")
    feat_extract = nn.Sequential(*list(resnet_model.children())[:-1])
    
    model = PhotoCaptioner(feat_extract, vocab_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epoch):
        print(f"\nEpoch {epoch+1}/{num_epoch}")
        train_one_epoch(model, train_loader, optimizer, device)

        test_loss, test_acc = evaluate(model, test_loader, device)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(f"Test Loss: {test_loss:.4f} | Token acc: {test_acc:.2f}%")
        print(f"Val  Loss: {val_loss:.4f} | Token acc: {val_acc:.2f}%")

if __name__ == "__main__":
    main()