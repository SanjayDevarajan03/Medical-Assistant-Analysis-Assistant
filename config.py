import os
from pathlib import Path

class Config:
    """Configuration settings for Medical Image Captioning with Attention."""

    # Project Path
    ROOT_DIR = Path(__file__).parent.parent.resolve()
    DATA_DIR = ROOT_DIR / "data"
    MODEL_DIR = ROOT_DIR / "checkpoints"
    OUTPUT_DIR = ROOT_DIR / "outputs"
    LOG_DIR = ROOT_DIR / "logs"

    # Create directories if they don't exist
    for dir_path in [DATA_DIR, MODEL_DIR, OUTPUT_DIR, LOG_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    # Dataset configuration
    DATASET = "IU-Xray"
    IU_XRAY_PATH = DATA_DIR/ "iu-xray"
    MIN_WORD_FREQ = 3
    MAX_CAPTION_LENGTH = 60
    TRAIN_VAL_TEST_SPLIT = [0.8, 0.1, 0.1]

    # Image processing
    IMG_SIZE= (224, 224)
    PIXEL_MEAN = [0.485, 0.456, 0.406]
    PIXEL_STD = [0.229, 0.224, 0.225]

    # Medical Image specific
    WINDOW_CENTER = 50 # Default window center for CT/X-ray (Hounsfield units)
    WINDOW_WIDTH = 350  # Default window width for CT/X-ray (Hounsfield units)
    USE_DICOM_WINDOWING = True  # Whether to use DICOM window settings when available

    # Model Architecture
    CNN_ENCODER = "densenet121"
    PRETRAINED = True
    FREEZE_CNN = False
    EMBEDDING_CNN = False
    HIDDEN_DIM = 512
    ATTENTION_DIM = 512
    DECODER_LAYERS = 6
    ATTENTION_HEADS = 8
    EMBEDDING_DIM = 512
    DROPOUT = 0.5

    # Training Parameters
    BATCH_SIZE = 132
    LEARNING_RATE = 3e-4 
    ENCODER_LR_FACTORY = 0.1 # Learning Rate Factor for encoder (for fine-tuning)
    WEIGHT_DECAY = 1e-5
    EPOCHS = 30
    CLIP_GRAD_NORM = 5.0 # Gradient clipping norm
    EARLY_STOPPING_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5 # Factor by which to reduce learning rate on plateau
    SCHEDULER_PATIENCE = 3 # Number of epochs with no improvement to wait before reducing LR


    # System Settings
    SEED = 42 # Random seed for reproducibility
    DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    NUM_WORKERS = 4
    LOG_INTERVAL = 100
    SAVE_INTERVAL = 1


    # Streamlit App settings
    APP_TITLE = "Medical Image Captioning"
    APP_DESCRIPTION = "Upload a medical image to generate a radiological description"
    ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".cdm"]