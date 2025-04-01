import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from pathlib import Path
import pydicom
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

from model import MedicalCaptionModel
from data_loader import Vocabulary, load_and_preprocess_image
from config import Config
import argparse

class MedicalCaptionPredictor:
    def __init__(self, model_path, vocab_path):
        """Initialize the predictor with model and vocabulary."""
        self.device = Config.DEVICE

        # load vocabulary
        self.vocab = Vocabulary()
        self.vocab.load(vocab_path)

        # Initialize model
        self.model = MedicalCaptionModel(vocab_size=len(self.vocab))

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize(Config.IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.PIXEL_MEAN, std=Config.PIXEL_STD)
        ])

    def predict(self, image_path, beam_size=3):
        """Generate cpation for an image using beam search"""
        # Load and preprocess image
        if isinstance(image_path, str) or isinstance(image_path, Path):
            image = load_and_preprocess_image(image_path)
        else:
            # Assuming it's a;ready a PIL Image
            image = image_path
        
        image = self.transform(image).unsqueeze(0).to(self.device)

        # Generate caption
        with torch.no_grad():
            if beam_size <= 1:
                # Greedy search
                output_seq, alphas = self.model.predict(image)
                output_seq = output_seq .squeeze(0).cpu().numpy()

                # Convert indices to words
                caption = []
                for idx in output_seq:
                    word = self.vocab.idx2word[idx.item()]
                    if word == "<EOS>":
                        break
                    if word not in ["<PAD>", "<SOS>"]:
                        caption.append(word)

                return " ".join(caption)
            else:
                # Beam Search
                return self._beam_search(image, beam_size)
            

    def _beam_search(self, image, beam_size=3, max_length=20):
        """Beam search for better caption generation"""
        # Get encoder output
        encoder_out = self.model.encoder(image)

        # Initialize LSTM state
        h,c = self.model.decoder.init_hidden_state(encoder_out)

        # We' ll keep track of the top beam_size candidates
        candidates = [([self.vocab.word2idx["<SOS>"]], 0.0, h,c)]

        # Beam Search
        for _ in range(max_length):
            temp_candidates = []

            # Expand each current candidate
            for seq, score, h, c in candidates:
                if seq[-1] == self.vocab.word2idx["<EOS>"]:
                    # This sequence is complete, keep it
                    temp_candidates.append((seq, score, h, c))
                    continue
                # Get last word embedding
                word = torch.tensor([seq[-1]]).to(self.device)
                emb = self.model.decoder.embedding(word)

                # Get attention-weighted encoding
                attention_weighted_encoding, alpha = self.model.decoder.attention(encoder_out, h)

                # Apply gating
                gate = torch.sigmoid(self.model.decoder.f_beta(h))
                attention_weighted_encoding = gate * attention_weighted_encoding

                # Update hidden state
                h_new, c_new = self.model.decoder.decode_step(
                    torch.cat([emb, attention_weighted_encoding], dim=1),
                    (h,c)
                )

                # Get prediction 
                preds = self.model.decoder.fc(h_new)

                # Get top k woirds
                log_probs = torch.log_softmax(preds, dim=1)
                top_k_probs, top_k_words = log_probs.topk(beam_size)

                # Create new candidates
                for i in range(beam_size):
                    next_word = top_k_words[0,i].item()
                    next_score = score + top_k_probs[0,i].item()
                    temp_candidates.append((seq+[next_word], next_score, h_new, c_new))

            # Select top beam_szie candidates
            candidates = sorted(temp_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

            # Check of all candidates end with <EOS>
            if all(c[0][-1] == self.vocab.word2idx["<EOS>"] for c in candidates):
                break

        # Get best sequence
        best_seq = candidates[0][0]

        # COnvert indices to words
        caption = []
        for idx in best_seq:
            word = self.vocab.idx2word[idx]
            if word == "<EOS>":
                break
            if word not in ["<PAD>", "<SOS>"]:
                caption.append(word)

        return " ".join(caption)
    

def visualize_attention(predictor, image_path, output_path=None):
    """Visualize attention maps for caption generation"""
    # Load and preprocess image
    original_image = load_and_preprocess_image(image_path)
    image = predictor.transform(original_image).unsqueeze(0).to(predictor.device)

    # Generate caption with attention
    with torch.no_grad():
        encoder_out = predictor.model.encoder(image)
        predictions, alphas = predictor.model.decoder.predict(encoder_out)

        # Convert predcitions to words
        predictions = predictions.squeeze(0).cpu().numpy()
        alphas = alphas.squeeze(0).cpu().numpy()

        caption = []
        for idx in predictions:
            word = predictor.vocab.idx2word[idx.item()]
            if word == "<EOS>":
                break
            if word not in ["<PAD>", "<SOS>"]:
                caption.append(word)


    # Visualize
    caption_text = " ".join(caption)
    num_words = len(caption)

    # Create a figure to display the image and attention maps
    plt.figure(figsize=(15,10))

    # Display the original image
    plt.subplot(num_words+1,2,1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis("Off")

    # Display caption
    plt.subplot(num_words+1,2,2)
    plt.text(0.5, 0.5, caption_text, fontsize=20, ha="center")
    plt.axis("off")

    # Display attention maps
    cmap = get_cmap("viridis")
    encoder_dim = encoder_out.size(1)
    attention_map_size = int(np.sqrt(encoder_dim))

    for i in range(num_words):
        # Rshape attention map
        word_attention = alphas[i].reshape(attention_map_size, attention_map_size)

        # Display attention map
        plt.subplot(num_words+1, 2, 2*i+3)
        plt.imshow(original_image)
        plt.imshow(word_attention, cmap=cmap, alpha = 0.7)
        plt.title(f"Attention: {caption[i]}")
        plt.axis("off")

    plt.tight_layout()

    if output_path:
        plt.save(output_path)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate captions for medical images")
    parser.add_argument("--image", type=str, required=True, help="Path to the image")
    parser.add_argument("--model", type=str, default=str(Config.MODEL_DIR/"best_model.pth"), help="Path to the model checkpoint")
    parser.add_argument("--vocab", type=str, default=str(Config.MODEL_DIR/"vocabulary.pth"), help = "Path to the vocabulary file")
    parser.add_argument("--beam_size", type=int, default=3, help="Beam size for beam search")
    parser.add_argument("--visualize", action = "store_true", help="Visualize attention maps")
    parser.add_argument("--output", type=str, help="Path to save visualization")

    args = parser.parse_args()

    # Initialize predictor
    predictor = MedicalCaptionPredictor(args.model, args.vocab)

    # Generate caption
    caption = predictor.predict(args.image, beam_size = args.beam_size)

    print(f"Caption: {caption}")

    # Visualize attention if requested
    if args.visualize:
        visualize_attention(predictor, args.image, args.output) 