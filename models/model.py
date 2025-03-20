import torch
import torch.nn as nn
import torch.nn.functional as models
from config import Config

class Encoder(nn.Module):
    def __init__(self, encoded_image_size=7):
        super(Encoder, self).__init__()


    # Load pretrained CNN
        if Config.CNN_ENCODER == "densenet121":
            cnn = models.densenet121(pretrained=True)
            self.enc_idm = 1024
            # Remove linear and pool layers
            modules = list(cnn.children())[:-1]
            self.cnn = nn.Sequential(*modules)

        elif Config.CNN_ENCODER == "resnet50":
            cnn = models.resnet50(pretrained=True)
            self.enc_idm = 2048

            # Remove linear and pool layers
            modules = list(cnn.children())[:-2]
            self.cnn = nn.Sequential(*modules)