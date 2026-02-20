import torch
import torch.nn as nn


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
    def __init__(self, encoder, vocab_size, pad_idx):
        """
        Initializes the PhotoCaptioner model.
        """
        super().__init__()
        self.encoder = encoder
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.projection = nn.Linear(2048, 512)
        self.embed = nn.Embedding(vocab_size, 512, padding_idx=0)

        self.decoder = nn.LSTM(input_size=512, hidden_size=512, num_layers=2, batch_first=True, dropout=0.2)
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