import torch
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from src.utils.data_loader import get_dataloaders
from src.model.photo_captioner import PhotoCaptioner


def train_one_epoch(model, loader, optimizer, device, pad_idx=0):
    """
    Trains the model for a single epoch for training dataset.

    Iterates over the dataloader, performs a forward pass with teacher forcing,
    computes cross-entropy loss, backpropagates, and updates model parameters.

    Args:
        model (nn.Module): The captioning model to train.
        loader (DataLoader): Dataloader providing (images, captions) batches. 
                             Captions are expected to be padded token index tensors.
        optimizer (torch.optim.Optimizer): Optimizer used to update model parameters.
        device (torch.device): Device to run training on (CPU or CUDA).
        pad_idx (optional): ignores the pad
    """
    model.train()
    loss_function = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.05)

    total_loss = 0.0
    correct = 0
    total = 0

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

        total_loss += loss.item()
        prediction = outputs.argmax(dim=-1)
        mask = targets != pad_idx

        correct += (prediction[mask] == targets[mask]).sum().item()
        total += mask.sum().item()

        loss.backward()
        optimizer.step()

        avg_loss = total_loss / len(loader)
        token_acc = 100.0 * correct / max(total, 1)
        if (i + 1) % 50 == 0:
            print(f"Batch {i+1:5d} training loss: {loss.item():.4f} | Token acc: {token_acc:.2f}%")

    print("Finished Training")
    return avg_loss, token_acc


def evaluate(model, loader, device, pad_idx=0):
    """
    Trains the model for a single epoch for test/validation dataset.

    Iterates over the dataloader, performs a forward pass with teacher forcing,
    computes cross-entropy loss, backpropagates, and updates model parameters.

    Args:
        model (nn.Module): The captioning model to train.
        loader (DataLoader): Dataloader providing (images, captions) batches. 
                             Captions are expected to be padded token index tensors.
        optimizer (torch.optim.Optimizer): Optimizer used to update model parameters.
        device (torch.device): Device to run training on (CPU or CUDA).
        pad_idx (optional): ignores the pad
    """
    model.eval()
    loss_function = nn.CrossEntropyLoss(ignore_index=0)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, captions in loader:
            images = images.to(device)
            captions = captions.to(device).long()

            captions_in = captions[:, :-1]
            targets = captions[:, 1:]

            outputs = model(images, captions_in)
            outputs = outputs[:, 1:, :]

            loss = loss_function(
                outputs.reshape(-1, outputs.size(-1)),
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--resume_ckpt", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    train_loader, test_loader, val_loader, vocab = get_dataloaders(batch_size=64, num_workers=4)
    vocab_size = len(vocab.word2idx)
    num_epoch = 10
    base_lr = 1e-5
    pad_idx = vocab.word2idx["<PAD>"]

    resnet_model = models.resnet50(weights="IMAGENET1K_V1")
    feat_extract = nn.Sequential(*list(resnet_model.children())[:-1])
    
    model = PhotoCaptioner(feat_extract, vocab_size, pad_idx).to(device)
    
    if args.resume_ckpt:
        model.load_state_dict(torch.load(args.resume_ckpt, map_location=device))

    if args.finetune:
        for p in model.encoder.parameters():
            p.requires_grad = False

        model.encoder[7].train()
        for p in model.encoder[7].parameters():
            p.requires_grad = True


    if args.finetune:
        optimizer = torch.optim.AdamW([
            {"params": list(model.encoder[7].parameters()), "lr": 5e-6},
            {"params": list(model.projection.parameters())
                    + list(model.embed.parameters())
                    + list(model.decoder.parameters())
                    + list(model.fc_out.parameters()), "lr": 6e-5},
        ], weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    best_val = float("inf")

    if args.finetune:
        model.encoder.eval()       # freeze BN running stats globally
        model.encoder[7].train()

    train_losses, val_losses = [], []
    train_accs, val_accs = [], []


    for epoch in range(num_epoch):
        print(f"\nEpoch {epoch+1}/{num_epoch}")

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device, pad_idx=pad_idx)
        val_loss, val_acc = evaluate(model, val_loader, device, pad_idx=pad_idx)

        train_losses.append(tr_loss); train_accs.append(tr_acc)
        val_losses.append(val_loss);   val_accs.append(val_acc)
        print(f"Train Loss: {tr_loss:.4f} | Train Token acc avg: {tr_acc:.2f}%")
        print(f"Val  Loss: {val_loss:.4f} | Token acc: {val_acc:.2f}%")

        scheduler.step(val_loss)
        if args.finetune:
            print("Encoder LR:", optimizer.param_groups[0]["lr"], "| Decoder LR:", optimizer.param_groups[1]["lr"])
        else:
            print("LR now:", optimizer.param_groups[0]["lr"])

        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), "best_v4_finetune.pt")

    metrics = pd.DataFrame({
        "epoch": list(range(1, num_epoch + 1)),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_token_acc": train_accs,
        "val_token_acc": val_accs,
    })
    metrics.to_csv("learning_curve.csv", index=False)
    # plot loss curve
    plt.figure()
    plt.plot(metrics["epoch"], metrics["train_loss"], label="Train Loss")
    plt.plot(metrics["epoch"], metrics["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Learning Curve (Loss)")
    plt.legend()
    plt.grid(True)
    plt.savefig("loss_curve.png", dpi=200)
    plt.show()

    # plot accuracy curve (optional)
    plt.figure()
    plt.plot(metrics["epoch"], metrics["train_token_acc"], label="Train Token Acc")
    plt.plot(metrics["epoch"], metrics["val_token_acc"], label="Val Token Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Token Accuracy (%)")
    plt.title("Learning Curve (Token Accuracy)")
    plt.legend()
    plt.grid(True)
    plt.savefig("acc_curve.png", dpi=200)
    plt.show()

    model.load_state_dict(torch.load("best_v4_finetune.pt", map_location=device))
    print("Model loaded")
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f} | Token acc: {test_acc:.2f}%")

if __name__ == "__main__":
    main()