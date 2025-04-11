import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class ImageDatasetWithContext(Dataset):
    def __init__(self, image_dir, json_file, transform=None):
        self.image_dir = image_dir
        self.transform = transform

        # Load metadata JSON
        with open(json_file, 'r') as f:
            metadata = json.load(f)
        # Extract images and annotations
        self.image_metadata = metadata["images"]  # List of image metadata dicts
        self.annotations = {ann["image_id"]: ann["category_id"] for ann in metadata["annotations"]}  # Map image_id to label
        # Create a mapping: {filename -> (latitude, longitude, label)}
        self.image_data = {}
        for img in self.image_metadata:
            filename = img["file_name"].split("/")[-1]
            image_id = img["id"]

            lat, long = img["latitude"], img["longitude"]
            if type(img["latitude"]) is not float or img["latitude"] > 180 or img["latitude"] < -180:
                lat = 0
            if type(img["longitude"]) is not float or img["longitude"] > 180 or img["longitude"] < -180:
                long = 0   

            # normalise to be in range of cnn output [-0.25, 4.5]
            lat = ((lat + 180) / 360) * 4.75 - 0.25
            long = ((long + 180) / 360) * 4.75 - 0.25

            month = int(img["date"][5:7])
            month = (month / 12) * 4.75 - 0.25
            # remap labels to start from 0 
            label = self.annotations[image_id] - 3111
            self.image_data[filename] = (lat, long, month, label)
        # Get all valid image filenames that exist in metadata and directory
        self.image_files = []
        for root, _, files in os.walk(image_dir):
            for file in files:
                if file in self.image_data:
                    self.image_files.append(str("/" + root + "/" + file).split("\\")[-1])



    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_path = self.image_files[idx]
        file_name = file_path.split("/")[-1]
        image_path = os.path.join(self.image_dir, file_path)

        # Load and transform image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Retrieve lat/long and category
        lat, long, month, label = self.image_data[file_name]

        # Convert to tensors
        context_features = torch.tensor([lat, long, month], dtype=torch.float32)

        return image, context_features, label