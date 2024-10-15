import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.target_transform = target_transform
        self.image_filenames = sorted(os.listdir(image_dir))
        self.label_filenames = {os.path.splitext(f)[0]: f for f in os.listdir(label_dir)}

        # Ensure that every image has a corresponding label
        self.image_filenames = [f for f in self.image_filenames if os.path.splitext(f)[0] in self.label_filenames]

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):

        image_filename = self.image_filenames[idx]
        label_filename = self.label_filenames[os.path.splitext(image_filename)[0]]



        image_path = os.path.join(self.image_dir, image_filename)
        label_path = os.path.join(self.label_dir, label_filename)



        image = Image.open(image_path).convert("RGB")  # Load image and ensure it's in RGB format
        label = Image.open(label_path).convert("L")  # Load label as a single channel (grayscale)

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)

        return image, label


