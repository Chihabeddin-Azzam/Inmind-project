import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
# Import your custom dataset class
from my_model.CustomDataset import CustomDataset
from my_model.my_cnn import My_cnn
from my_model.Loss import Loss


# Define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def custom_collate(batch):
    images = []
    labels = []
    for b in batch:
        images.append(b[0])  # Image tensor
        labels.append(b[1])  # Label tensor
    return images, labels


# Define hyperparameters
learning_rate = 0.001
num_epochs = 10
batch_size = 32

# Define transformations
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

# Create dataset
dataset = CustomDataset("yolov7/split_dataset_new/train/images", "yolov7/split_dataset_new/train/labels", transform=transform)

# Create data loader
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

# Create model instance
model = My_cnn(num_classes=3).to(device)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = Loss()

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(data_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
        # Move images and labels to the device
        images = torch.stack(images).to(device)
        # Determine the maximum number of bounding boxes in the batch
        max_boxes = max(len(label) for label in labels)

        # Pad the labels to ensure they have the same size
        padded_labels = []
        for label in labels:
            num_boxes_to_pad = max_boxes - len(label)
            padded_label = torch.cat([label, torch.zeros(num_boxes_to_pad, label.shape[1])])
            padded_labels.append(padded_label)

        # Convert the padded labels to a tensor and move to the device
        labels = torch.stack(padded_labels).to(device)


        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataset)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}')

print('Training Finished!')
