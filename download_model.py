import os
import requests

MODEL_URL = "https://huggingface.co/Mridul-2305/brain-tumor-model/resolve/main/vit_unet.pth"
MODEL_PATH = "vit_unet.pth"

def download_model():
    if os.path.exists(MODEL_PATH):
        print("Model already exists")
        return

    print("Downloading model...")

    response = requests.get(MODEL_URL, stream=True)

    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print("Download completed")