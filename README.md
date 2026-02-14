<h1 align="center">Photo Captioning Bot</h1>
<p align="center">
  <strong>Offline CNN–LSTM image captioning in PyTorch<br>
  Local. Private. Fast.</strong>
</p>
<p align="center">
  An offline image captioning model built with PyTorch that generates short, relevant captions for photos using a CNN–LSTM pipeline.  
  Designed as a local, privacy-friendly baseline for multimodal learning and experimentation.
</p>
<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.10-orange" />
  <img src="https://img.shields.io/badge/Python-3.13.7-blue" />
  <img src="https://img.shields.io/badge/Status-Educational-green" />
</p>

## Table of Contents
- [Security](#security)
- [Background](#background)
- [Install](#install)
- [Usage](#usage)
- [API](#api)
- [Contributing](#contributing)
- [License](#license)


## Security
This project is designed to run fully locally and doesnt uplaod images or captions to external servers to protect user data. All photo processing and caption generation happens on the user's machine to minimize privacy risks. Since the tool will operate on unencrypted local backups or personal photo directiores, users should ensure that sensitive data is handled securiy and the generated outputs are stored appropriately. This project is inteded for research and educational use.


## Background
Photo captioning sits in between computer vision and natural language processing where it enables machines to translate visual content into human readable text. While there are many solutions that exist, they rely on cloud APIs and closed systems, this project focuses on building a fully local CNN-LSTM captioning pipleine in PyTorch so users can keep their data private.

## Install
```bash
git clone https://github.com/yourname/photo-captioner.git  
cd photo-captioner
pip install -r requirements.txt
```
## Usage
**Training**
```bash
python -m src.training.train
```
**Inference**
```bash
python -m src.scripts.generate_caption \
  --image data/Images/[your_image.jpg] \
  --ckpt src/checkpoint/best.pt \
  --captions_csv data/captions.csv
```
*NOTE:* Change '[your_image.jpg]' to the name of your .jpg file without the brackets.  
**Example**
```bash
python -m src.scripts.generate_caption \
  --image data/Images/10815824_2997e03d76.jpg \
  --ckpt src/checkpoint/best.pt \
  --captions_csv data/captions.csv
```
**Output**  
```bash
Caption: a man in a red shirt is standing in front of a large crowd
```

## API
### Model
#### `PhotoCaptioner`
**Location:** `src/model/photo_captioning.py`

**Description:**  
Image captioning model that combines a CNN encoder with an LSTM decoder. The encoder extracts visual features from an image, which are projected into the embedding space and used by the decoder to generate a caption token by token.

**Constructor:**
```python
PhotoCaptioner(encoder, vocab_size, finetune_encoder=False)
...
```

### Data Loading

#### `get_dataloaders(...)`
**Location:** `src/utils/data_loader.py`

**Description:**  
Creates PyTorch DataLoaders for the Flickr8k training, validation, and test splits, including image preprocessing and caption tokenization.

**Function:**
```python
def get_dataloaders(batch_size=64, num_workers=2):
...
```
### Training
#### `train_one_epoch(...)`
**Location:** `src/training/train.py`

**Description:**  
Runs a single training epoch over a dataloader and updates model parameters.

**Function:**
```python
def train_one_epoch(model, loader, optimizer, device, pad_idx=0):
...
```

### Caption Generation

#### `generate_caption(...)`
**Location:** `src/scripts/generate_caption.py`

**Description:**  
Generates a caption for a single image using a trained captioning model.

**Function:**
```python
def generate_caption(model, image, vocab, max_len=30):
...
```

## Contributing
Contributions are welcome!
1. Fork the repo
2. Create a feature branch
3. Submit a PR with a clear description

## License
[MIT © Richard McRichface.](./LICENSE)
