import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from tqdm import tqdm

from models.model import MedicalCaptionModel
from data_loader import create_data_loaders, Vocabulary
from config import Config

def train():
    """Main training function"""
    # Set random seeds for replacibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    # Create model directory
    os.makedirs(Config.MODEL_DIR, exist_ok=True)

    # Create data loaders
    train_loader, val_loader = create_data_loaders()

    # Create vocabulary from the dataset
    vocab_size = len(train_loader.dataset.vocab)

    # Initialize model
    model = MedicalCaptionModel(vocab_size=vocab_size)
    model = model.to(Config.DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # Different learning rates for encoder and decoder
    encoder_params = list(model.encoder.parameters())
    decoder_params = list(model.decoder.parameters())

    optimizer = optim.Adam(
        [
            {'params': encoder_params, 'lr': Config.LEARNING_RATE* Config.ENCODER_LR_FACTORY},
            {'params': decoder_params}
        ],
        lr = Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor = Config.SCHEDULER_FACTOR,
        patience = Config.SCHEDULER_PATIENCE, verbose=True
    )

    # Initialize Tensorboard writer
    writer = SummaryWriter(Config.LOG_DIR)

    # Training variables
    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time.time()

    # Main training loop
    for epoch in range(Config.EPOCHS):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, criterion, epoch)

        # Evaluate on validation set
        val_loss = validate(model, val_loader, criterion)

        # Update learning rate
        scheduler.step(val_loss)

        # Print epoch statistics
        print(f'Epoch {epoch+1}/{Config.EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss)

        # Save model checkpoint
        if (epoch+1) % Config.SAVE_INTERVAL == 0:
            save_checkpoint(model, optimizer, epoch, val_loss, False)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_loss, True)



def train_epoch(model, data_loader, optimizer, criterion, epoch):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    batch_count = 0

    # Progress bar

