import argparse
import torch
from PIL import Image
import torchvision.transforms as T
from src.training.train import PhotoCaptioner
from src.utils.text import build_vocab_from_csv 
from torchvision import models
import torch.nn as nn
from src.utils.text import load_vocab



DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 30
SOS = "<SOS>"
EOS = "<EOS>"


def load_image(image_path):
    """
    Docstring for load_image
    
    :param image_path: Description
    """
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(image_path).convert("RGB")
    return transform(img).unsqueeze(0)


@torch.no_grad()
def generate_caption(model, image, vocab, max_len=30):
    """
    
    """
    model.eval()
    device = next(model.parameters()).device
    image = image.to(device)

    # Encode image
    feature = model.encoder(image)
    feature = torch.flatten(feature, 1)              # [1, 2048]
    projected = model.projection(feature)            # [1, 512]

    sos_idx = vocab.word2idx["<SOS>"]
    eos_idx = vocab.word2idx["<EOS>"]
    unk_idx = vocab.word2idx.get("<UNK>", None)

    generated = []
    prev_token = torch.tensor([[sos_idx]], device=device)  # [1, 1]

    hidden = None

    for _ in range(max_len):
        emb = model.embed(prev_token)                 # [1, 1, 512]

        projected_step = projected.unsqueeze(1)       # [1, 1, 512]
        x = torch.cat([projected_step, emb], dim=1)   # [1, 2, 512]

        outputs, hidden = model.decoder(x, hidden)
        logits = model.fc_out(outputs[:, -1, :])
        if unk_idx is not None:
            logits[0, unk_idx] = -1e9
        next_id = logits.argmax(dim=-1).item()

        if next_id == eos_idx:
            break

        generated.append(next_id)
        prev_token = torch.tensor([[next_id]], device=device)

    return " ".join(vocab.idx2word[i] for i in generated)


def build_model(vocab_size, pad_idx):
    """
    Docstring for build_model
    
    :param vocab_size: Description
    """
    resnet = models.resnet50(weights="IMAGENET1K_V1")
    resnet.fc = nn.Identity()
    encoder = nn.Sequential(*list(resnet.children())[:-1])
    model = PhotoCaptioner(encoder, vocab_size=vocab_size, pad_idx=pad_idx)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--ckpt", type=str, default="best.pt", help="Path to model checkpoint")
    parser.add_argument("--captions_csv", type=str, required=True, help="CSV used to build vocab")
    args = parser.parse_args()

    vocab = load_vocab("vocab.pkl")
    vocab_size = len(vocab.word2idx)

    pad_idx = vocab.word2idx["<PAD>"]
    model = build_model(vocab_size, pad_idx).to(DEVICE)
    model.load_state_dict(torch.load(args.ckpt, map_location=DEVICE))

    image = load_image(args.image)

    caption = generate_caption(model, image, vocab, max_len=MAX_LEN)
    print("\nImage:", args.image)
    print("Caption:", caption)


if __name__ == "__main__":
    main()