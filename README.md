# Image Search with CLIP and DINO

This project implements an image search application using CLIP and DINO models. It allows users to search for images using text queries and provides functionality to export results.

## Features

- **Image Search**: Search for images using text queries.
- **Model Support**: Utilizes CLIP and DINO models for image similarity search.
- **FAISS Index**: Supports generating and loading FAISS indices for efficient search.
- **Export Functionality**: Export selected images and search results to CSV files.

## Streamlit Test App

![alt text](media/CLIP-DINOv2-POC.gif)

<!-- For full working video of streamlit app : 
`github.com/artzuros/clip-dinov2-nlq/blob/master/media/CLIP-DINOv2-POC.mp4` -->
## Requirements

- Python 3.9+
- CUDA for GPU acceleration (optional but recommended for better performance)
- miniconda or Conda installed
## Installation

Follow these steps to set up the project:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/artzuros/clip-dinov2-nlq.git
   cd clip-dinov2-nlq
2. **Create conda environment**
    ```bash
    conda create -n clip-dinov2-nlq python=3.10.4 # Use any version > 3.9
    conda activate clip-dinov2-nlq
3. **Installing Dependencies**
    ```bash
    pip install -r requirements.txt # Torch with CPU
    ```
    for GPU acceleration
    ```bash
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 # Change according to your CUDA Drivers
    pip install -r requirements_gpu.txt # Rest of the dependencies
4. **Download Models** : 
    Make sure the necessary models are available. This code will automatically download the models on first use if they are not already present.

### USAGE

To run the application, run the following command:
```bash
streamlit run test.py
```
### Follow these steps to use the application:

- Enter the Directory Path: Provide the path to the directory containing your images.
- Generate FAISS Index: Click the "Generate FAISS Index" button to create the index for your images.
- Load FAISS Index: If you have already generated the index, you can load it using the "Load FAISS Index" button.
- Enter Search Query: Type in your text query to search for similar images.
- Select Images: Use the slider to control the number of images to display (increments of 20).
- Export Selected Images: Export selected images to a CSV file using the provided button.
- Process with DINO: Optionally process selected images with the DINO model for further similarity searches.

### Structure
---
```
├── test.py                   # Main Streamlit app
├── utils
│   ├── utils.py             # Utility functions
├── requirements.txt         # Python dependencies
├── requirements_gpu.txt     # Python dependencies for GPU
├── dinov2-clip.ipynb        # Testing Jupyter notebook
└── README.md                # Project documentation
```