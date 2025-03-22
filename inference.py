import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import pydicom

from models.model import MedicalCaptionModel
from data_loader import Vocabulary, load_and_preprocess_image
from config import Config

class MedicalCaptionPredictor:
    def __init__(self, model_path, vocab_path):
        """Initialize the predictor with model and vocabulary."""
        self.device = Config.DEVICE

        # load vocabulary
        self.vocab = Vocabulary()
        self.vocab.load(vocab_path)

        # Initialize odel
        self.model = MedicalCaptionModel(vocab_size=len(self.vocab))

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Image transforms
        self.transfrom = transforms.Compose([
            transforms.Resize(Config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.PIXEL_MEAN, std=Config.PIXEL_STD)
        ])

    def predict(self, image_path, beam_size=3):
        
