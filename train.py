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
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}")

    for i, (images, captions) in enumerate(progress_bar):
        # Move to device
        images = images.to(Config.DEVICE)
        captions = captions.to(Config.DEVICE)

        # Calculate caption lengths
        caption_lengths =torch.tensor([len(cap)-cap.eq(0).sum().item() for cap in captions])
        caption_lengths = caption_lengths.unsqueeze(1).to(Config.DEVICE)

        # Zero gradients
        optimizer.zero_grad()


        # Forward pass
        predictions, encoded_captions, decode_lengths, alphas, sort_ind = model(images, captions, caption_lengths)

        # Calculate loss
        targets = encoded_captions[:,1:] # Remove <SOS>

        # Pack predictions for variable length sequences
        predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

        # Calculate loss
        loss = criterion(predictions, targets)

        # Add only doubly stochastic attention regularization
        if Config.ATTENTION_DIM > 0:
            alpha_loss = ((1-alphas.sum(dim=1))**2).mean()
            loss += alpha_loss * 0.01

        # Backward pass
        loss.backward()

        # Clip gradients
        if Config.CLIP_GRAD_NORM > 0:
            nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)

        # Update weights
        optimizer.step()

        # Update progress bar
        progress_bar.set_postfix(loss=loss.item())

        # Update statistics
        epoch_loss += loss.item()
        batch_count += 1

        # log interval
        if (i+1)% Config.LOG_INTERVAL == 0:
            print(f"Epoch {epoch+1}, Batch {i+1}/{len(data_loader)}, Loss: {loss.item():.4f}")

    return epoch_loss/batch_count

def validate(model, data_loader, criterion):
    """Evaluate model on validation set"""
    model.eval()
    val_loss = 0
    batch_count = 0

    with torch.no_grad():
        for images, captions in tqdm(data_loader, desc="Validation"):
            # Move to device
            images = images.to(Config.DEVICE)
            captions = captions.to(Config.DEVICE)

            # Calculate caption lengths
            caption_lengths = torch.tensor([len(cap)-cap.eq(0).sum().item() for cap in captions])
            caption_lengths = caption_lengths.unsqueeze(1).to(Config.DEVICE)

            # Forward pass
            predictions, encoded_captions, decode_lengths, alphas, sort_ind = model(images, captions, caption_lengths)

            # Calculate loss 
            targets = encoded_captions[:,1:]

            # Pack predictions for variable length sequences
            predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            # Calculate loss
            loss = criterion(predictions, targets)

            # Update statistics
            val_loss += loss.item()
            batch_count += 1

    return val_loss / batch_count

def save_checkpoint(model, optimizer, epoch, val_loss, is_best=False, is_final=False):
    """Save model checkpoint"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state.dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss
    }

    if is_best:
        checkpoint_path = os.path.join(Config.MODEL_DIR, "best_model.pth")
    elif is_final:
        checkpoint_path = os.path.join(Config.MODEL_DIR, "final_model.pth")
    else:
        checkpoint_path = os.path.join(Config.MODEL_DIR, f"model_epoch_{epoch+1}.pth")

    torch.save(checkpoint,checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")


if __name__ == "__main__":
    train()
