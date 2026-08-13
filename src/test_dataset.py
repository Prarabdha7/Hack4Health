import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import MedicalImageDataset


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = MedicalImageDataset(
    "data",
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

images, labels = next(iter(loader))

print("Number of images:", len(dataset))
print("Classes:", dataset.classes)
print("Class mapping:", dataset.class_to_idx)
print("Batch shape:", images.shape)
print("Labels:", labels)