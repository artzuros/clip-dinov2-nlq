import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import faiss

from transformers import AutoModel, AutoImageProcessor
from transformers import CLIPModel, AutoProcessor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

processor_dino = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
model_dino = AutoModel.from_pretrained('facebook/dinov2-base').to(device)

processor_clip = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
model_clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)

class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_path

def normalizeL2(embeddings):
    vector = embeddings.detach().cpu().numpy()
    vector = np.float32(vector)
    faiss.normalize_L2(vector)
    return vector

def add_vector_to_index(embedding, index):
    vector = embedding.detach().cpu().numpy()
    vector = np.float32(vector)
    faiss.normalize_L2(vector)
    index.add(vector)

def extract_features_dino(image):
    with torch.no_grad():
        inputs = processor_dino(images=image, return_tensors="pt").to(device)
        outputs = model_dino(**inputs)
        image_features = outputs.last_hidden_state
        return image_features.mean(dim=1)

def extract_features_clip(image):
    with torch.no_grad():
        inputs = processor_clip(images=image, return_tensors="pt").to(device)
        image_features = model_clip.get_image_features(**inputs)
        return image_features
    
def extract_features_clip_text(text):
    with torch.no_grad():
        inputs = processor_clip(text=text, return_tensors="pt").to(device)
        text_features = model_clip.get_text_features(**inputs)
        return text_features
    
def search_index_text(text_query, index_clip):
    with torch.no_grad():
        inputs = processor_clip(text=text_query, return_tensors="pt").to(device)
        text_features = model_clip.get_text_features(**inputs)
    
    input_features_clip = normalizeL2(text_features)
    
    D_clip_text, I_clip_text = index_clip.search(input_features_clip, 10)
    
    return D_clip_text, I_clip_text

def search_index_image(image, index_dino):
    dino_features = extract_features_dino(image)
    dino_features = normalizeL2(dino_features)
    D_dino, I_dino = index_dino.search(dino_features, 10)
    
    return D_dino, I_dino
    