import re
from collections import Counter


def tokenize(text):
    """
    Tokenizes and normalizes a caption string.

    This function lowercases the input text and extract word tokens using
    a simple regex-based tokenizer.

    Args:
        text (str): Captions or sentence to tokenize

    Returns:
        list: List of lowercase words tokens extracted from input text.
    """
    return re.findall(r"\b\w+\b", text.lower())

class Vocabulary:
    """
    Vocabulary class for mapping tokens to integer indices and back.

    This class builds a word-to-index and index-to-word mapping from a list
    of tokens, filtering out rare words below a frequency threshold.

    Special tokens:
        <PAD>: Padding token (index 0)
        <UNK>: Unknown token
        <SOS>: Start-of-sequence token
        <EOS>: End-of-sequence token

    Args:
        tokens (list): List of tokens used to build the vocabulary.
        threshold (int, optional): Minimum frequency required for a token to be
                                   included in the vocabulary. Defaults to 2.
    """
    def __init__(self, tokens, threshold=2):
        self.threshold = threshold

        self.word2idx = {"<PAD>":0, "UNK":1, "<SOS>":2, "<EOS>":3}
        self.idx2word = {i:w for w, i in self.padding.items()}
        self.idx = 4

        self.freq = Counter(" ".join(tokens).split())

        for word, count in self.freq.items():
            if count >= self.threshold:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def token_to_id(self, token):
        return self.word2idx.get(token, self.word2idx["<UNK>"])

    def id_to_token(self, idx):
        return self.idx2word.get(idx, "<UNK>")