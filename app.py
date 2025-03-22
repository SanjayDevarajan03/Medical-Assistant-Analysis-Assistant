import streamlit as st
import os
from PIL import Image
import torch
import tempfile
from models.encoder import EncoderCNN
from models.decoder import DecoderTransformer
from inference.caption_generator import generate_caption
from config import Config

# Set page configuration
st.set_age_config(
    page_title = Config.APP_TITLE,
    page_icon = "",
    layout = "wide"
)

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
