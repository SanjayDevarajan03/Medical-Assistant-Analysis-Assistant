import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pydicom
from collections import Counter
from pathlib import Path
from config import Config

class Vocabulart:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3:"<UNK"}
        self.word_count = Counter()
        self.idx = 4 # next index to assign

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1


    def __len__(self):
        return len(self.word2idx)
    
    def build_vocab(self, report_texts, threshold=3):
        """Build vocabulary from report texts"""
        word_counts = Counter()
        for text in report_texts:
            word_counts.update(text.lower().split())

        for word, count in word_counts.items():
            if count >= threshold:
                self.add_word(word)

        print(f"Vocabulary built with {len(self)} words")

    def save(self, path):
        torch.save({'word2idx': self.word2idx, 'idx2word': self.idx2word, 'idx': self.idx}, path)

    def load(self, path):
        data = torch.load(path)
        self.word2idx = data['word2idx']
        self.idx2word = data['idx2word']
        self.idx = data['idx']


def load_and_preprocess_image(image_path):
    """Load and process medical image"""
    path = Path(image_path)

    # Handle Dicom files
    if path.suffix.lower() == '.dcm':
        try:
            dicom = pydicom.dmread(path)
            image_array = dicom.pixel_array

            # Apply windowing if available
            if hasattr(dicom, 'WindowCenter') and hasattr(dicom, 'WindowWidth'):
                window_center = dicom.WindowCenter
                window_width  = dicom.WindowWidth
                if hasattr(window_center, '__iter__'):
                    window_center = window_center[0]
                if hasattr(window_width, '__iter__'):
                    window_width = window_width[0]
                if hasattr(window_width, '__iter__'):
                    window_width = window_width[0]

            else:
                window_center = Config.WINDOW_CENTER

        except:
            pass



            # Apply windowing if available



            

