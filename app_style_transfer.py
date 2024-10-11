# Import necessary libraries
import functools
import altair as alt
import numpy as np
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray

# Load TensorFlow Hub module for arbitrary image stylization
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)

# Print TensorFlow and TF-Hub versions, and check for GPU availability
print("TF Version: ", tf.__version__)
print("TF-Hub version: ", hub.__version__)
print("Eager mode enabled: ", tf.executing_eagerly())
print("GPU available: ", tf.config.list_physical_devices('GPU'))

# Function to crop the center of an image to make it square
def crop_center(image):
    """Returns a cropped square image."""
    shape = image.shape
    new_shape = min(shape[0], shape[1])
    offset_y = max(shape[0] - shape[1], 0) // 2
    offset_x = max(shape[1] - shape[0], 0) // 2
    image = tf.image.crop_to_bounding_box(
        image, offset_y, offset_x, new_shape, new_shape)
    return image

# Function to load and preprocess images
def load_images(uploaded_files, image_size=(256, 256)):
    images = []
    for uploaded_file in uploaded_files:
        img = Image.open(uploaded_file)
        img = tf.convert_to_tensor(img)
        img = crop_center(img)
        img = tf.image.resize(img, image_size)
        if img.shape[-1] == 4:
            img = img[:, :, :3]  # Remove alpha channel if present
        img = tf.reshape(img, [-1, image_size[0], image_size[1], 3]) / 255
        images.append(img)
    return images

# Function to create a collage from a list of images
def create_collage(images, image_size=(256, 256), layout='horizontal'):
    num_images = len(images)

    if layout == 'horizontal':
        cols = num_images
        rows = 1
    elif layout == 'vertical':
        cols = 1
        rows = num_images

    collage_width = cols * image_size[0]
    collage_height = rows * image_size[1]

    collage = np.zeros((collage_height, collage_width, 3))
    index = 0
    for i in range(rows):
        for j in range(cols):
            if index < num_images:
                img = images[index][0].numpy()
                collage[i * image_size[0]:(i + 1) * image_size[0],
                        j * image_size[1]:(j + 1) * image_size[1], :] = img
                index += 1
    return tf.convert_to_tensor(collage[np.newaxis, ...], dtype=tf.float32)

# Function to display an image using Streamlit
def show_image(image, col=st):
    col.image(np.array(image[0]))

# Function to resize an image to a target size
def resize_image(image, target_size):
    """Resize image to the target size."""
    image = tf.image.resize(image, target_size)
    return image

# Function to calculate the Structural Similarity Index (SSIM) between two images
def calculate_ssim(image1, image2):
    """Calculate the Structural Similarity Index (SSIM) between two images and return the score as a percentage."""
    image1 = np.array(image1[0].numpy())
    image2 = np.array(image2[0].numpy())

    # Convert images to grayscale for SSIM comparison
    image1_gray = np.mean(image1, axis=-1)
    image2_gray = np.mean(image2, axis=-1)

    # Resize images to the same dimensions
    min_height = min(image1_gray.shape[0], image2_gray.shape[0])
    min_width = min(image1_gray.shape[1], image2_gray.shape[1])
    image1_gray = tf.image.resize(image1_gray[..., np.newaxis], (min_height, min_width)).numpy().squeeze()
    image2_gray = tf.image.resize(image2_gray[..., np.newaxis], (min_height, min_width)).numpy().squeeze()

    ssim_score = ssim(image1_gray, image2_gray, data_range=image2_gray.max() - image2_gray.min())
    
    # Convert SSIM score to percentage
    ssim_percentage = ssim_score * 100
    return ssim_percentage

# Function to calculate color similarity between two images
def calculate_color_similarity(image1, image2):
    """Calculate color similarity between two images as a percentage based on histogram matching."""
    # Convert images to numpy arrays
    image1 = np.array(image1[0].numpy())
    image2 = np.array(image2[0].numpy())

    # Convert images to grayscale for histogram comparison
    image1_gray = rgb2gray(image1)
    image2_gray = rgb2gray(image2)

    # Calculate histograms for the grayscale images
    hist1, _ = np.histogram(image1_gray, bins=256, range=(0, 1))
    hist2, _ = np.histogram(image2_gray, bins=256, range=(0, 1))

    # Normalize histograms
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)

    # Calculate histogram correlation
    correlation = np.corrcoef(hist1, hist2)[0, 1]
    
    # Convert correlation to percentage
    color_similarity_percentage = correlation * 100
    return color_similarity_percentage

## Basic setup and app layout
st.set_page_config(layout="wide")

# Set Altair renderer options
alt.renderers.set_embed_options(scaleFactor=2)

# Hide Streamlit style elements for a cleaner look
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Main function
if __name__ == "__main__":
    img_width, img_height = 256, 256

    # Create columns for content and style images
    col1, col2 = st.columns(2)
    col1.markdown('# CONTENT IMAGES')

    # Slider to select the number of images for the collage
    num_images = col1.slider("Select the number of images for the collage", min_value=1, max_value=10, value=4)
    # File uploader to upload content images
    uploaded_files = col1.file_uploader("Choose images to create a collage", accept_multiple_files=True)

    # Radio buttons to select the layout for the collage
    layout = col1.radio("Select the layout for the collage", ['horizontal', 'vertical'])

    if uploaded_files:
        # Ensure the correct number of images are uploaded
        if len(uploaded_files) < num_images:
            st.warning(f"Please upload {num_images - len(uploaded_files)} more content images to meet the required number.")
        elif len(uploaded_files) > num_images:
            st.warning(f"You have uploaded {len(uploaded_files)} images. Only the first {num_images} images will be used.")
            uploaded_files = uploaded_files[:num_images]

        # Load and display content images
        content_images = load_images(uploaded_files, (img_width, img_height))
        collage_image = create_collage(content_images, (img_width, img_height), layout)
        col1.image(np.array(collage_image[0]), caption='Collage Image')

    col2.markdown('# STYLE IMAGES')

    # Slider to select the number of style images
    num_styles = col2.slider("Select the number of styles to apply", min_value=1, max_value=5, value=1)
    # File uploader to upload style images
    uploaded_files_style = col2.file_uploader("Choose style images", accept_multiple_files=True)

    if uploaded_files_style:
        # Ensure the correct number of style images are uploaded
        if len(uploaded_files_style) > num_styles:
            st.warning(f"You have uploaded {len(uploaded_files_style)} style images. Only the first {num_styles} will be used.")
            uploaded_files_style = uploaded_files_style[:num_styles]

        # Load and display style images
        style_images = load_images(uploaded_files_style, (img_width, img_height))
        for i, style_image in enumerate(style_images):
            col2.image(np.array(style_image[0]), caption=f'Style Image {i+1}')
        
        # Process and display styled collages
        styled_collages = []
        for style_image in style_images:
            style_image = tf.nn.avg_pool(style_image, ksize=[3, 3], strides=[1, 1], padding='SAME')
            outputs = hub_module(tf.constant(collage_image), tf.constant(style_image))
            styled_collages.append(outputs[0])

        # Create columns for displaying styled collages
        col3, col4, col5 = st.columns(3)
        col3.markdown('# Styled Collages')
        for i, (styled_collage, style_image) in enumerate(zip(styled_collages, style_images)):
            col3.image(np.array(styled_collage[0]), caption=f'Styled Collage {i+1}')
            
            # Calculate and display color similarity
            similarity_percentage = calculate_color_similarity(style_image, styled_collage)
            col3.markdown(f"Similarity with Style Image {i+1}: {similarity_percentage:.2f}%")
