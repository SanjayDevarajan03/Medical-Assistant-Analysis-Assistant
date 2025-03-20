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
        else:
            raise ValueError(f"Unsupported CNN encoder: {Config.CNN_ENCODER}")
        
        # Resize output to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))


        # ?
        self.fine_tune()


    def forward(self, images):
        """
        Forward propogation
        images: [batch_size, 3, height, width]
        """
        features = self.cnn(images) # [batch_size, enc_dim, feat_h, feat_w]
        features = self.adaptive_pool(features) # [batch_size, enc_dim, enc_img_size]
        features = features.permute(0,2,3,1) # [batch_size, enc_img_size, enc_img_size, enc_dim]

        # Flatten spatial dimensions
        batch_size = features.size(0)
        features = features.view(batch_size, -1, self.enc_idm) # [batch_size, num_pixels, enc_dim]