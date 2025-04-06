import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

# Define the transformations for the training and test datasets
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Define the paths to the ImageNetLT dataset
data_dir = './data/ImageNetLT'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'val')

# Check if the dataset directories exist
if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    raise FileNotFoundError(f"ImageNetLT dataset not found in {data_dir}. Please ensure the dataset is downloaded and extracted.")

# Load the training dataset
trainset = datasets.ImageFolder(root=train_dir, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

# Load the test dataset
testset = datasets.ImageFolder(root=test_dir, transform=transform_test)
testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# Print the number of training and test samples
print(f'Number of training samples: {len(trainset)}')
print(f'Number of test samples: {len(testset)}')

# Example: Iterate through the training data
for images, labels in trainloader:
    print(f"Batch size: {images.size(0)}, Image shape: {images.size()}, Labels: {labels}")
    break