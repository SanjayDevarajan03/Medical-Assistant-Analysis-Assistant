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

class Vocabulary:
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
                window_width = Config.WINDOW_WIDTH
            

            # Apply windowing
            window_min = window_center - window_width // 2
            window_max= window_center + window_width // 2
            image_array = np.clip(image_array, window_min, window_max)
            image_array = (image_array - window_min)/(window_max-window_min)

            # Convert to RGB
            image_array = np.stack([image_array, image_array, image_array], axis=2)
            image_array = (image_array * 255).astype(np.uint8)
            image = Image.fromarray(image_array)
        except Exception as e:
            print(f"Error reading DICOM: {e}")

    
    else:
        # Handle regular image formats
        try:
            image = Image.open(path).convert('RGB')
        except:
            print(f"Error printing image: {path}")
            return None
        
    
    return image


class MedicalCaptionDataset(Dataset):
    def __init__(self, image_paths, captions, vocabulary, transform=None):
        self.image_paths = image_paths
        self.captions= captions
        self.vocab = vocabulary
        self.transform = transform or transforms.Compose([
            transforms.Resize(Config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.PIXEL_MEAN, std=Config.PIXEL_STD)
        ])


    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption = self.captions[idx]

        # Load and preprocess image
        image = load_and_preprocess_image(image_path)
        if image is None:
            # Return a placeholder if image loading fails
            image = Image.new('RGB', Config.IMG_SIZE, color=0)

        if self.transform:
            image = self.transform(image)


        # Tokenize caption 
        tokens = [self.vocab.word2idx.get(word.lower(), self.vocab.word2idx["<UNK>"]) for word in caption.split()]
        tokens = [self.vocab.word2idx["<SSO>"]] + tokens + [self.vocab.word2idx["<EOS>"]]

        # Pad or truncate caption
        if len(tokens) < Config.MAX_CAPTION_LENGTH:
            tokens = tokens + [self.vocab.word2idx["<PAD>"]] * (Config.MAX_CAPTION_LENGTH)
        else:
            tokens = tokens[:Config.MAX_CAPTION_LENGTH-1] + [self.vocab.word2idx["<EOS>"]]

        return image, torch.tensor(tokens)

def get_iu_xray_data():
    """
    Process IU X-Ray dataset - this is a placeholder function
    You'll need to adapt this to your specific dataset structure
    """
    # This is where you'd parse your specific dataset files
    # For example, loading the IU X-ray reports and extracting relevant sections

    # Placeholder  - replace with actual data loading
    image_paths = []
    captions = []


def create_data_loaders():
    """Create train and validation data loaders"""
    # Get data
    image_paths, captions = get_iu_xray_data()

    # Create vocabulary
    vocab = Vocabulary()
    vocab.build_voacb(captions, threshold = Config.MIN_WORD_FREQ)

    # Split data
    data_size= len(image_paths)
    indices = list(range(data_size))
    np.random.shuffle(indices)

    split = int(0.9 * data_size)
    train_indices = indices[:split]
    val_indices = indices[split:]


    # Creata datasets
    train_transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.2), # Careful with medical images
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.PIXEL_MEAN, std = Config.PIXEL_STD)     
    ]) 

    val_transform = transforms.Compose([
        transforms.Resize(Config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=Config.PIXEL_MEAN, std = Config.PIXEL_STD)
    ])

    train_dataset = MedicalCaptionDataset(
        [image_paths[i] for i in train_indices],
        [captions[i] for i in train_indices],
        vocab,
        transform = train_transform
    )

    val_dataset = MedicalCaptionDataset
    



            # Apply windowing if available



            

