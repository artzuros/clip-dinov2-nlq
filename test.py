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
    
    # Create dataset and data loader
    dataset = ImageDataset(images, transform=transform)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    # Build FAISS index
    index_clip = faiss.IndexFlatL2(512)
    index_dino = faiss.IndexFlatL2(768)

    for image in images:
        img = Image.open(image).convert('RGB')
        clip_features = extract_features_clip(img)
        add_vector_to_index(clip_features, index_clip)

        dino_features = extract_features_dino(img)
        add_vector_to_index(dino_features, index_dino)

    faiss.write_index(index_clip, 'index_clip.index')
    faiss.write_index(index_dino, 'index_dino.index')

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
                # st.write(f"Image path: {image_path}")
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

    # Use selected images for further image search using DINO
    # dino_features = []
    # for image_path in st.session_state.selected_images:
    #     img = Image.open(image_path).convert('RGB')
    #     dino_features.append(extract_features_dino(img))

    # dino_features = torch.stack(dino_features)
    # dino_features = normalizeL2(dino_features)
    
    # Perform DINO image search
    image_from_clip = Image.open(st.session_state.selected_images[0]).convert('RGB')
    D_dino, I_dino = search_index_image(image_from_clip, index_dino)

    st.write("Top matching images using DINO:")
    for idx in range(len(st.session_state.selected_images)):
        st.write(f"Selected Image {idx + 1}:")
        for i in range(10):
            image_path = images[I_dino[idx][i]]
            img = Image.open(image_path)
            st.image(img, use_column_width=True)
            st.write(f"Distance: {D_dino[idx][i]:.3f}")
    
    # Select DINO Images and export them to csn (query, selected_image, similar_image, distance)
    selected_dino_images = []
    for idx in range(len(st.session_state.selected_images)):
        selected_dino_images.append(images[I_dino[idx][0]])

    st.write("Selected DINO Images:")
    for selected_image in selected_dino_images:
        st.image(selected_image, use_column_width=True)

    # Export the selected images to CSV
    if st.button("Export to CSV"):
        import pandas as pd

        data = []
        for idx, selected_image in enumerate(st.session_state.selected_images):
            data.append([query, selected_image, selected_dino_images[idx], D_dino[idx][0]])

        df = pd.DataFrame(data, columns=["Query", "Selected Image", "Similar Image", "Distance"])
        df.to_csv("selected_images.csv", index=False)
