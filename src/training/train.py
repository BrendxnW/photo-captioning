import torchvision.models as models
import torch.nn as nn



resnet_model = models.resnet18(weights="IMAGENET1K_V1")
feat_extract = nn.Sequential(*list(resnet_model.children())[:-1])