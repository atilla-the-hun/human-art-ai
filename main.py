import streamlit as st
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image
import os
import uuid
import base64
import re

from generate_image import generate_image

load_dotenv()

st.set_page_config(
    page_title="Human Art AI",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure the images directory exists
IMAGES_DIR = "generated_images"
UPLOADED_DIR = "uploaded_images"
if not os.path.exists(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)
if not os.path.exists(UPLOADED_DIR):
    os.makedirs(UPLOADED_DIR)

def save_image(image, format="PNG", directory=IMAGES_DIR, source_filename=None):
    """Save image to disk and return the filename"""
    if format.upper() == 'JPG':
        format = 'JPEG'
    filename = f"{uuid.uuid4()}.{format.lower()}"
    filepath = os.path.join(directory, filename)
    image.save(filepath, format=format.upper())
    
    # Save metadata about the source image
    if source_filename:
        metadata_filename = f"{filename}.meta"
        metadata_filepath = os.path.join(directory, metadata_filename)
        with open(metadata_filepath, 'w') as f:
            f.write(source_filename)
    
    return filename

def get_aspect_ratio(image):
    """Determine the aspect ratio of an image"""
    width, height = image.size
    ratio = width / height
    
    if 1.9 <= ratio < 2.3:
        return "21:9"
    elif 1.5 <= ratio < 1.9:
        return "16:9"
    elif 1.25 <= ratio < 1.5:
        return "3:2"
    elif 0.9 <= ratio < 1.25:
        return "1:1"
    elif 0.75 <= ratio < 0.9:
        return "4:5"
    elif 0.6 <= ratio < 0.75:
        return "2:3"
    elif 0.4 <= ratio < 0.6:
        return "9:16"
    else:
        return "9:21"

def load_images():
    """Load all images from the images directory"""
    images = {}
    for filename in os.listdir(IMAGES_DIR):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            filepath = os.path.join(IMAGES_DIR, filename)
            img = Image.open(filepath)
            aspect_ratio = get_aspect_ratio(img)
            
            # Read metadata
            metadata_filename = f"{filename}.meta"
            metadata_filepath = os.path.join(IMAGES_DIR, metadata_filename)
            if os.path.exists(metadata_filepath):
                with open(metadata_filepath, 'r') as f:
                    source_filename = f.read().strip()
            else:
                source_filename = "Unknown"
            
            if aspect_ratio not in images:
                images[aspect_ratio] = {}
            if source_filename not in images[aspect_ratio]:
                images[aspect_ratio][source_filename] = []
            images[aspect_ratio][source_filename].append((filename, img))
    return images

def image_to_base64(image):
    """Convert image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# Streamlit app
# Load the image you want to display next to the title
image_path = "./brain_boost_2.png"  # Replace with the actual path to your image
image = Image.open(image_path)

# Create a custom HTML layout for the title and image
st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: flex-start;">
        <div style="display: flex; flex-direction: column; align-items: flex-start;">
            <h1 style="margin-top: -24px; margin-right: 10px;">Human Art AI</h1>
            <h3 style="margin-top: -10px; margin-right: 10px;"><a href="https://brain-boost-ai-pros.vercel.app" target="_blank" style="text-decoration: none; color: inherit;">by Brain Boost</a></h3>
        </div>
        <a href="https://brain-boost-ai-pros.vercel.app" target="_blank">
            <img src="data:image/png;base64,{}" width="50" style="margin-left: 10px;" />
        </a>
    </div>
    """.format(image_to_base64(image)),
    unsafe_allow_html=True
)

st.subheader("Transform yourself into any character with a prompt")

# Sidebar
st.sidebar.title("Settings")

# Move the file uploader to the top of the sidebar
uploaded_file = st.sidebar.file_uploader("Upload Main Face Image", type=["png", "jpg", "jpeg", "webp"])

# Add dropdown boxes and sliders in the sidebar
num_outputs = st.sidebar.number_input("Number of Outputs", value=1, min_value=1, max_value=4, step=1)
output_format = st.sidebar.selectbox("Output Format", ["png", "webp"], index=0)

# Add sliders for height and width
width = st.sidebar.slider("Width (pixels)", min_value=256, max_value=1536, value=896, step=1)
height = st.sidebar.slider("Height (pixels)", min_value=256, max_value=1536, value=1152, step=1)

# New input parameters
true_cfg = st.sidebar.slider("True CFG", min_value=1.0, max_value=10.0, value=1.0, step=0.01)
id_weight = st.sidebar.slider("ID Weight", min_value=1.0, max_value=3.0, value=1.05, step=0.01)
num_steps = st.sidebar.slider("Number of Steps", min_value=1, max_value=20, value=20, step=1)
start_step = st.sidebar.slider("Start Step", min_value=0, max_value=10, value=0, step=1)
guidance_scale = st.sidebar.slider("Guidance Scale", min_value=1.0, max_value=10.0, value=4.0, step=0.01)
negative_prompt = st.sidebar.text_area("Negative Prompt", value="bad quality, worst quality, text, signature, watermark, extra limbs, low resolution, partially rendered objects, deformed or partially rendered eyes, deformed, deformed eyeballs, cross-eyed, blurry")
max_sequence_length = st.sidebar.slider("Max Sequence Length", min_value=128, max_value=512, value=128, step=1)

# Add Reset button to the sidebar
if st.sidebar.button("Reset"):
    # Clear session state
    st.session_state.clear()
    st.sidebar.success("Page has been reset. Your generated images are still available in the showcase.")

# Main content area
prompt = st.text_input("Enter a prompt:", value=" ")

# Generate Image button
if st.button("Generate Image", disabled=uploaded_file is None):
    if uploaded_file is not None:
        with st.spinner("Uploading and processing image..."):
            # Read the uploaded file
            image = Image.open(uploaded_file)
            
            # Save the uploaded image
            filename = save_image(image, format=image.format, directory=UPLOADED_DIR)
            
            # Get the full path of the saved image
            main_face_image_path = os.path.join(UPLOADED_DIR, filename)
            
            with st.spinner("Generating image..."):
                try:
                    generated_images, elapsed_time = generate_image(
                        prompt, num_outputs, output_format, width, height, main_face_image_path,
                        true_cfg=true_cfg, id_weight=id_weight, num_steps=num_steps, 
                        start_step=start_step, guidance_scale=guidance_scale, 
                        negative_prompt=negative_prompt, max_sequence_length=max_sequence_length
                    )
                    st.write(f"Image generated in {elapsed_time:.2f} seconds")
                    
                    # Save generated images to disk and update session state
                    st.session_state.full_size_images = []
                    for img in generated_images:
                        filename = save_image(img, format=output_format, source_filename=uploaded_file.name)
                        st.session_state.full_size_images.append((img, filename))
                except Exception as e:
                    st.error(f"Error generating image: {e}")
    else:
        st.warning("Please upload a main face image.")

# Automatically display the full-size images if new ones were generated
if st.session_state.get('full_size_images', []):
    for img, filename in st.session_state.full_size_images:
        st.image(img, use_column_width=True)
        # Add download button for the full-size image
        img_byte_arr = BytesIO()
        save_format = output_format.upper()
        img.save(img_byte_arr, format=save_format)
        img_byte_arr = img_byte_arr.getvalue()

        st.download_button(
            label="Download Full Size Image",
            data=img_byte_arr,
            file_name=filename,
            mime=f"image/{output_format}"
        )
else:
    st.write("No images generated yet.")

# Display images by aspect ratio and source image
st.subheader("Image Showcase")
saved_images = load_images()
has_images = any(images for images in saved_images.values())

if has_images:
    for aspect_ratio, source_images in saved_images.items():
        st.subheader(f"Aspect Ratio: {aspect_ratio}")
        for source_filename, images in source_images.items():
            st.write(f"Source: {source_filename}")
            columns = st.columns(4)  # Create 4 columns for the grid
            for i, (filename, img) in enumerate(images):
                with columns[i % 4]:
                    # Create a unique key for each image
                    key = f"img_{aspect_ratio}_{source_filename}_{i}"
                    # Display thumbnail without the fullscreen hover icon
                    st.image(img, width=100, use_column_width=True)
                    # Create a button that when clicked will open the full-size image
                    button_col = st.columns([1, 2, 1])  # Create three columns for centering
                    with button_col[1]:  # Use the middle column
                        # Add download button for the full-size image
                        img_byte_arr = BytesIO()
                        save_format = 'JPEG' if filename.lower().endswith('.jpg') else filename.split('.')[-1].upper()
                        img.save(img_byte_arr, format=save_format)
                        img_byte_arr = img_byte_arr.getvalue()

else:
    st.write("No images generated yet.")

# Adding custom CSS to remove fullscreen icon on hover and move buttons slightly to the right
st.markdown(
    """
    <style>
    /* Remove fullscreen icon when hovering over images */
    button[title="View fullscreen"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)