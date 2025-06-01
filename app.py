import streamlit as st
import os
import torch
from os.path import join

from PIL import Image

from utils import get_model
from generate import run
import tempfile


IMAGES_PATH = join("images")
MODEL_NAME   = "en3FR_lstm3_aug1_ncap1#0"
# DEVICE       = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
DEVICE       = torch.device('cpu')


@st.cache_data
def get_image_paths():
    images = [join(IMAGES_PATH, path) for path in os.listdir(IMAGES_PATH) if path.endswith(".png")]
    return images


@st.cache_data
def load_image(path):
    return Image.open(path)


@st.cache_resource
def load_model(model_name):
    model = get_model("chkp", model_name, DEVICE)
    model = model.to(DEVICE)
    model.eval()
    return model


def place_image(paths, title):
    st.header(title)
    st.image([load_image(path) for path in paths], width=200)


def welcome_block(model):
    image_paths = get_image_paths()

    start, end = 0, 3
    place_image(image_paths[start:end], title="Work with any image")

    if st.button("Transform Example"):
        st.header("Transformed Examples")
        col1, col2, col3 = st.columns(3)
        with st.spinner("Applying style transfer..."):
            try:
                results = [
                    run(model, image_paths[i])
                    for i in range(end)
                ]

                # Display
                with col1:
                    st.image(results[0][0], caption=results[0][1], use_container_width=True)
                with col2:
                    st.image(results[1][0], caption=results[1][1], use_container_width=True)
                with col3:
                    st.image(results[2][0], caption=results[2][1], use_container_width=True)

            except Exception as e:
                st.error(f"Error during inference: {str(e)}")


def inference_block(model):
    st.write("Upload an image to get a caption!")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=False
    )

    if uploaded_file is not None:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(uploaded_file.getvalue())
            input_path = tmp.name

        # Result placeholder
        col, = st.columns(1)

        # Process and inference
        if st.button("Transform!"):
            with st.spinner("Applying style transfer..."):
                try:
                    image, text = run(model, input_path)

                    # Display
                    with col:
                        st.image(image, caption=text, use_container_width=True)

                except Exception as e:
                    st.error(f"Error during inference: {str(e)}")
                finally:
                    os.unlink(input_path)  # Cleanup temp file

if __name__ == "__main__":
    model = load_model(MODEL_NAME)
    welcome_block(model)
    inference_block(model)
