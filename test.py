import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from transformers import CLIPModel, AutoProcessor
import faiss
import os
import numpy as np
import datetime
from torchvision import transforms
import streamlit as st
import pandas as pd

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

    faiss.write_index(index_clip, DIR_PATH + '/' + 'index_clip.index')
    faiss.write_index(index_dino, DIR_PATH + '/' + 'index_dino.index')

# Load FAISS index
def load_faiss_index():
    if os.path.exists(DIR_PATH + '/' + 'index_clip.index') and os.path.exists(DIR_PATH + '/' + 'index_dino.index'):
        index_clip = faiss.read_index(DIR_PATH + '/' + 'index_clip.index')
        index_dino = faiss.read_index(DIR_PATH + '/' + 'index_dino.index')
        return index_clip, index_dino
    else:
        st.error("FAISS index files not found!")
        return None, None

# Streamlit buttons
if st.button("Generate FAISS Index"):
    index_clip = faiss.IndexFlatL2(512)
    index_dino = faiss.IndexFlatL2(768)
    generate_faiss_index(images, index_clip, index_dino)
    st.session_state.index_clip = index_clip
    st.session_state.index_dino = index_dino
    st.success("FAISS index generated and saved successfully!")

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

    # Provide options for export or further processing after CLIP selection
    if "selected_images" in st.session_state and st.session_state.selected_images:
        if st.button("Export Selected Images to CSV"):
            data = []
            for image_path, distance in zip(st.session_state.selected_images, st.session_state.selected_distances):
                data.append([query, image_path, distance])

            df = pd.DataFrame(data, columns=["Query", "Image", "Distance"])
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            df.to_csv(DIR_PATH + '/' + timestamp + "selected_images_clip.csv", index=False)
            st.success("Selected images exported to CSV!")

        if st.button("Process Selected Images with DINO"):
            st.session_state.dino_search_results = []

            for selected_image in st.session_state.selected_images:
                img = Image.open(selected_image).convert('RGB')
                D_dino, I_dino = search_index_image(img, index_dino)
                for d, i in zip(D_dino[0], I_dino[0]):
                    st.session_state.dino_search_results.append((selected_image, images[i], d))

            # Get the 20 images with the least distances
            st.session_state.dino_search_results.sort(key=lambda x: x[2])
            st.session_state.dino_search_results = st.session_state.dino_search_results[:20]

            st.write("Top matching images using DINO:")
            cols = st.columns(5)
            for idx, (selected_image, similar_image, distance) in enumerate(st.session_state.dino_search_results):
                col = cols[idx % 5]
                with col:
                    img = Image.open(similar_image)
                    st.image(img, use_column_width=True)
                    st.write(f"Distance: {distance:.3f}")

            st.session_state.dino_search_export = True

        if st.session_state.get("dino_search_export", False):
            if st.button("Export DINO Results to CSV"):
                data = []
                for query_image, similar_image, distance in st.session_state.dino_search_results:
                    data.append([query, query_image, similar_image, distance])

                df = pd.DataFrame(data, columns=["Query", "Selected Image", "Similar Image", "Distance"])
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                df.to_csv(DIR_PATH + '/' + timestamp + "selected_images_dino.csv", index=False)
                st.success("DINO results exported to CSV!")

else:
    st.warning("Please load or generate the FAISS index first.")