import os
import json
import torch
import pandas as pd
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
    def __init__(self, image_paths, captions, vocabulary, transform=None, cache_size=1000):
        self.image_paths = image_paths
        self.captions= captions
        self.vocab = vocabulary
        self.transform = transform or transforms.Compose([
            transforms.Resize(Config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.PIXEL_MEAN, std=Config.PIXEL_STD)
        ])

        # Simple caching mechanism
        self.image_cache = {}
        self.cache_size = cache_size




    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption = self.captions[idx]

        # Check cache first
        if image_path in self.image_cache:
            image = self.image_cache[image_path]
        else:
            # Load and preprocess image
            image = load_and_preprocess_image(image_path)

            if len(self.image_cache) >= self.cache_size:
                # Remove oldest item
                self.image_cache.pop(next(iter(self.image_cache)))
            self.image_cache[image_path] = image

        
        if image is None:
            # Return a placeholder if image loading fails
            image = Image.new('RGB', Config.IMG_SIZE, color=0)

        if self.transform:
            image = self.transform(image)


        # Tokenize caption 
        tokens = [self.vocab.word2idx.get(word.lower(), self.vocab.word2idx["<UNK>"]) for word in caption.split()]
        tokens = [self.vocab.word2idx["<SOS>"]] + tokens + [self.vocab.word2idx["<EOS>"]]

        # Pad or truncate caption
        if len(tokens) < Config.MAX_CAPTION_LENGTH:
            tokens = tokens + [self.vocab.word2idx["<PAD>"]] * (Config.MAX_CAPTION_LENGTH)
        else:
            tokens = tokens[:Config.MAX_CAPTION_LENGTH-1] + [self.vocab.word2idx["<EOS>"]]

        return image, torch.tensor(tokens)
    
    def __len__(self):
        """Return the total number of samples in the dataset."""
        # If your dataset is based on a list of samples, return its length
        return len(self.image_paths)  # Adjust this to match your actual data structure

def get_iu_xray_data():
    """
    Process IU X-Ray dataset - this is a placeholder function
    You'll need to adapt this to your specific dataset structure
    """
    # This is where you'd parse your specific dataset files
    # For example, loading the IU X-ray reports and extracting relevant sections

    reports_path = Config.REPORTS_PATH
    projections_path = Config.PROJECTIONS_PATH
    if not reports_path.exists() or not projections_path.exists():
        print(f"Warning: CSV files not found.")
    image_paths = []
    captions = []

    
    

    # Sample data processing logic (replace with actual implementation)
    dataset_path = Config.IU_XRAY_PATH
    images_dir = dataset_path/"images"/"images_normalized"
    reports_file = dataset_path/"reports.json"

    # Check if paths exist and use dummy data if they don't
    if not images_dir.exists() or not reports_file.exists():
        print(f"Warning: Data path not found. Using dummy data instead.")
        print(f"Images directory: {images_dir}")
        print(f"Reports file: {reports_file}")
        
        # Create dummy data for testing - UNCOMMENT THIS SECTION
        for i in range(100):
            image_paths.append(f"dummy_path_{i}.jpg")
            captions.append(f"This is a normal chest X-ray with no significant findings.")
        
        print(f"Created {len(image_paths)} dummy samples")
        return image_paths, captions
    
    # If we get here, both paths exist
    # Load reports
    with open(reports_file, 'r') as f:
        reports = json.load(f)

    # Process reports and match with images
    for study_id, report in reports.items():
        study_dir = images_dir / study_id

        # Check if study directory exists
        if not study_dir.exists():
            continue

        # Find all .dcm and .png files in the study directory
        study_images = list(study_dir.glob("*.dcm")) + list(study_dir.glob("*.png"))

        # Extract findings
        findings = report.get("findings", "No findings recorded")

        # Add each image with the study's report
        for img_path in study_images:
            image_paths.append(str(img_path))
            captions.append(findings)

    # If no images were found, use dummy data
    if not image_paths:
        print("No valid images found. Using dummy data.")
        for i in range(100):
            image_paths.append(f"dummy_path_{i}.jpg")
            captions.append(f"This is a normal chest X-ray with no significant findings.")

    print(f"Loaded {len(image_paths)} images with corresponding reports")
    return image_paths, captions



def create_data_loaders():
    """Create train and validation data loaders"""
    # Get data
    image_paths, captions = get_iu_xray_data()

    # Create vocabulary
    vocab = Vocabulary()
    vocab.build_vocab(captions, threshold = Config.MIN_WORD_FREQ)

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

    val_dataset = MedicalCaptionDataset(
        [image_paths[i] for i in val_indices],
        [captions[i] for i in val_indices],
        vocab,
        transform = train_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers = Config.NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS
    )

    return train_loader, val_loader


            

