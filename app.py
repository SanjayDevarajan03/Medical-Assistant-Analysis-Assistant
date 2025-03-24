import streamlit as st
import os
from PIL import Image
import torch
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from inference  import visualize_attention, MedicalCaptionPredictor
from config import Config

# Set page configuration
st.set_age_config(
    page_title = Config.APP_TITLE,
    page_icon = "",
    layout = "wide"
)

@st.cache_resource
def load_model():
    """Load model and vocabulary (cached for performance)"""
    model_path = os.path.join(Config.MODEL_DIR, "best_model.pth")
    vocab_path = os.path.join(Config.MODEL_DIR, "vocabulary.pth")

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.info("Please train the model first or provide the correct path.")
        return None
    
    if not os.path.exists(vocab_path):
        st.error(f"Vocabulary file not found: {vocab_path}")
        st.info("Please train the model first or provide the correct path.")
        return None
    
    try:
        return MedicalCaptionPredictor(model_path, vocab_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    

def generate_caption(predictor, image, beam_size=3):
    """Generate caption for the uploaded image"""
    if predictor is None:
        return "Model not loaded"
    
    try:
        caption = predictor.predict(image,)
    except Exception as e:
        st.error(f"Error generating caption: {e}")
        return "Error generating caption"
    
def create_attention_visualization(predictor, image):
    """Create attention visualization for the image."""
    if predictor is None:
        return None
    
    try:
        # Process image
        img_tensor = predictor.transform(image).unsqueeze(0).to(predictor.device)

        # Generate caption with attention
        with torch.no_grad():
            encoder_out = predictor.model.encoder(img_tensor)
            predictions,alphas = predictor.model.decoder.predict(encoder_out)

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

            


def main():
    st.title("Medical Image Captioning")
    st.write("Upload a mdeical image (X-ray, MRI, CT Scan) to generate a description")

    uploaded_file = st.file_uploader("Uploaded Medical Image", use_column_width=True)
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Medical Image", use_column_width=True)

        # Generate caption
        if st.button("Generate caption"):
            with st.spinner("Analyzing image ..."):
                caption, attention__weights = generate_caption(image)

            st.success("Analysis Complete")
            st.write("### Generated Report:")
            st.write(caption)

            # Visualize Attention
            st.write("### Attention Visualization")
            attention_img = visualize_attention(image, caption, attention__weights)
            st.image(attention_img, use_column_width=True)

if __name__ == "__main__":
    main()
