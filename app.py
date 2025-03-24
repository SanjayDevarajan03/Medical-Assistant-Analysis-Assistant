import streamlit as st
import os
from PIL import Image
import torch
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from inference  import visualize_attention, MedicalCaptionPredictor
from data_loader import load_and_preprocess_image
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

        # Create figure
        caption_text = " ".join(caption)
        num_words = min(len(caption), 6) # Limit to 6 words for display

        fig, axes = plt.subplot(2, num_words,  figsize=(15,5))

        # Reshape attention maps
        encoder_dim = encoder_out.size(1)
        attention_map_size = int(np.sqrt(encoder_dim))

        for i in range(num_words):
            # Reshape attention map
            word_attention = alphas[i].reshape(attention_map_size, attention_map_size)

            # Display attention map
            axes[0,i].imshow(word_attention, cmap = "viridis")
            axes[0,i].set_title(caption[i])
            axes[0,i].axis("off")

            # Display image with attention overlay
            axes[1,i].imshow(image)
            axes[i,i].imshow(word_attention,cmap="viridis", alphas=0.7)
            axes[1,i].set_title(f"Attention: {caption[i]}")
            axes[1,i].axis('off')

        plt.tight_layot()
        return fig, caption_text

    except Exception as e:
        st.error(f"Error creating visualization: {e}")
        return None, None

            


def main():
    st.title(Config.APP_TITLE)
    st.markdown(Config.APP_DESCRIPTION)

    # Load model
    predictor = load_model()

    # Sidebar
    st.sidebar.title("Settings")
    beam_size = st.sidebar.slider("Beam Size", min_value = 1, max_value=5, value=3, help="Larger values prvide better caption quality but slower inference")
    show_attention = st.sidebar.checkbox("Show attention Visualization", value=True, help= "Visualize which parts of the image the model focuses on")

    # Image upload
    st.header("Upload Medical Image")
    uploaded_file = st.file_uplaoder("Choose a medical image file", type=["png", "jpg", "jpeg", "dcm"], help = "Suppored formats: JPEG, PNG, DICOM")

    if uploaded_file is None:
        with st.spinner("Process image..."):
            try:
                file_extension = os.path.splitext(uploaded_file.name)[1].lower()
                if file_extension == ".dcm":
                    # For DICOM files, we need to save to a temp file first
                    with tempfile.NamedTemporaryFile(suffix = ".dcm", delete=False) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name
                    
                    # Use the function from data_loader
                    image = load_and_preprocess_image(tmp_path)
                    os.unlink(tmp_path) # Clena up temp file

                else:
                    # Regular image file
                    image = Image.open(uploaded_file).convert('RGB')
                
                # Create columns for layout
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("Uploaded Image")
                    st.image(image, caption="Uploaded medical image", use_column_width=True)

                # Generate caption
                caption = generate_caption(predictor, image, beam_size)

                with col2:
                    st.subheader("Generate Report")
                    st.markdown(f"**Description:**")
                    st.write(caption)

                # Show attention visualization
                if show_attention and predictor is not None:
                    st.subheader("Attention Visualization")
                    st.markdown("This shows whoch parts of the image the model focuses on for each word.")

                    fig, full_caption  =create_attention_visualization(predictor, image)
                    if fig:
                        st.pyplot(fig)

            except Exception as e:
                st.error(f"Error processing imageL {e}")


    # About Section
    st.sidebar.markdown("---")
    st.sidebar.header("About")
    st.sidebar.info(
        "This application uses a deep learning model to generate descriptive reports for "
        "for medical images. The model employs an encoder-decoder architecture with"
        "attention mechanism to focus on relevant parts of the image while generating text."
    )

    # Example images
    st.sidebar.markdown("---")
    st.sidebar.header("Example Images")

    example_dir = Path(Config.DATA_DIR)/"examples"
    if example_dir.exists():
        example_files = list(example_dir.glob("*.jpg")) + list(example_dir.glob("*.png"))
        if example_files:
            selected_example = st.sidebar.selectbox(
                "Choose an example image",
                options=example_files,
                format_func = lambda x: x.name
            )

            if st.sidebar.button("Load Example"):
                with st.spinner("Loading example..."):
                    image = Image.open(selected_example).convert('RGB')

                    # Create columns for layout
                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Example Image")
                        st.image(image, caption=f"Example: {selected_example.name}", use_column_width=True)

                    # Show attention visualization
                    if show_attention  and predictor is not None:
                        st.subheader("Attention Visualization")
                        st.markdown("This shows which parts of the image the model focuses on for each world.")

                        fig, full_caption - create_attention_visualization(predictor,image)
                        if fig:
                            st.pyplot(fig)


if __name__ == "__main__":
    main()





    """uploaded_file = st.file_uploader("Uploaded Medical Image", use_column_width=True)
    
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
            st.image(attention_img, use_column_width=True)"""

