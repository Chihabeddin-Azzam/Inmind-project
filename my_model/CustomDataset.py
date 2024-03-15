import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_dir, label_dir, input_size=(640, 640), transform=None):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.input_size = input_size
        self.transform = transform
        
        # List image and label files
        self.image_files = sorted(os.listdir(image_dir))
        self.label_files = sorted(os.listdir(label_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        label_name = os.path.join(self.label_dir, self.label_files[idx])

        # Load image
        image = Image.open(img_name).convert("RGB")

        # Load label (parse YOLO format label)
        label = self.parse_yolo_label(label_name)

        # Resize image
        image = image.resize(self.input_size)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label

    def parse_yolo_label(self, label_file):
        # Parse YOLO format label file to extract bounding box coordinates and class IDs
        with open(label_file, "r") as f:
            lines = f.readlines()

        # Extract bounding box coordinates and class IDs
        boxes = []
        for line in lines:
            # Each line in YOLO format: class_id, x_center, y_center, width, height
            class_id, x_center, y_center, width, height = map(float, line.split())
            
            # Convert YOLO format to bounding box coordinates (x1, y1, x2, y2)
            x1 = int((x_center - width / 2) * self.input_size[0])
            y1 = int((y_center - height / 2) * self.input_size[1])
            x2 = int((x_center + width / 2) * self.input_size[0])
            y2 = int((y_center + height / 2) * self.input_size[1])

            boxes.append([x1, y1, x2, y2, int(class_id)])

        return torch.tensor(boxes)


