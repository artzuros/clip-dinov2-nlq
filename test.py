import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from transformers import CLIPModel, AutoProcessor
import faiss
import os
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity

from utils.utils import ImageDataset, normalizeL2, add_vector_to_index, extract_features_dino, extract_features_clip, search_index_image, search_index_text

transform = transforms.Compose([
    transforms.ToTensor()
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models and processors
processor_dino = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model_dino = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

processor_clip = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

# Collect all images
images = []

DIR_PATH = st.text_input("Enter the directory path:")

if DIR_PATH:
    for root, dirs, files in os.walk(DIR_PATH):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                images.append(os.path.join(root, file))

# Function to generate FAISS index
def generate_faiss_index(images, index_clip, index_dino):
    for image in images:
        img = Image.open(image).convert('RGB')
        clip_features = extract_features_clip(img)
        add_vector_to_index(clip_features, index_clip)

        dino_features = extract_features_dino(img)
        add_vector_to_index(dino_features, index_dino)

    faiss.write_index(index_clip, 'index_clip.index')
    faiss.write_index(index_dino, 'index_dino.index')

# Load FAISS index
def load_faiss_index():
    if os.path.exists('index_clip.index') and os.path.exists('index_dino.index'):
        index_clip = faiss.read_index('index_clip.index')
        index_dino = faiss.read_index('index_dino.index')
        return index_clip, index_dino
    else:
        st.error("FAISS index files not found!")
        return None, None
col_load, col_generate = st.columns(2)
    # Streamlit buttons
with col_load:
    if st.button("Generate FAISS Index"):
        index_clip = faiss.IndexFlatL2(512)
        index_dino = faiss.IndexFlatL2(768)
        generate_faiss_index(images, index_clip, index_dino)
        st.session_state.index_clip = index_clip
        st.session_state.index_dino = index_dino
        st.success("FAISS index generated and saved successfully!")

with col_generate:
    if st.button("Load FAISS Index"):
        index_clip, index_dino = load_faiss_index()
        if index_clip is not None and index_dino is not None:
            st.session_state.index_clip = index_clip
            st.session_state.index_dino = index_dino
            st.success("FAISS index loaded successfully!")

# Ensure the indices are defined before performing search
if 'index_clip' in st.session_state and 'index_dino' in st.session_state:
    index_clip = st.session_state.index_clip
    index_dino = st.session_state.index_dino

    # Streamlit interface
    query = st.text_input("Enter your search query:")

    if query:
        # Perform CLIP text search
        D_clip_text, I_clip_text = search_index_text(query, index_clip)
        
        # Slider to control the number of images to display
        total_images_to_show = st.slider("Total images to show", 10, min(len(images), 200), 10, 10)

        st.session_state.selected_images = st.session_state.get("selected_images", [])
        st.session_state.selected_distances = st.session_state.get("selected_distances", [])

        cols = st.columns(5)  # Adjust the number of columns as needed

        for idx in range(total_images_to_show):
            image_path = images[I_clip_text[0][idx]]
            img = Image.open(image_path)
            col = cols[idx % 5]
            
            with col:
                st.image(img, use_column_width=True)
                st.write(f"Distance: {D_clip_text[0][idx]:.3f}")
                
                if st.checkbox(f"Select", key=f"select_{idx}"):
                    if image_path not in st.session_state.selected_images:
                        st.session_state.selected_images.append(image_path)
                        st.session_state.selected_distances.append(D_clip_text[0][idx])
                else:
                    if image_path in st.session_state.selected_images:
                        index = st.session_state.selected_images.index(image_path)
                        st.session_state.selected_images.pop(index)
                        st.session_state.selected_distances.pop(index)

    # Display the selected images
    if "selected_images" in st.session_state and st.session_state.selected_images:
        st.write("Selected Images:")
        for selected_image, distance in zip(st.session_state.selected_images, st.session_state.selected_distances):
            st.image(selected_image, use_column_width=True)
            st.write(f"Distance: {distance}")

        # Perform DINO image search for each selected image
        all_distances = []
        all_indices = []

        for selected_image in st.session_state.selected_images:
            img = Image.open(selected_image).convert('RGB')
            D_dino, I_dino = search_index_image(img, index_dino)
            all_distances.extend(D_dino[0])
            all_indices.extend(I_dino[0])

        # Get the 20 images with the least distances
        least_distance_indices = np.argsort(all_distances)[:20]
        top_images = [images[all_indices[idx]] for idx in least_distance_indices]
        top_distances = [all_distances[idx] for idx in least_distance_indices]

        st.write("Top matching images using DINO:")
        cols = st.columns(5)
        for idx in range(20):
            img_path = top_images[idx]
            col = cols[idx % 5]
            with col:
                img = Image.open(img_path)
                st.image(img, use_column_width=True)
                st.write(f"Distance: {top_distances[idx]:.3f}")

        # Export the selected images to CSV
        if st.button("Export to CSV"):
            import pandas as pd

            data = []
            for idx, selected_image in enumerate(st.session_state.selected_images):
                data.append([query, selected_image, top_images[idx], top_distances[idx]])

            df = pd.DataFrame(data, columns=["Query", "Selected Image", "Similar Image", "Distance"])
            df.to_csv("selected_images.csv", index=False)
else:
    st.warning("Please load or generate the FAISS index first.")
