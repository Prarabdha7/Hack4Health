import os

from PIL import Image
from torch.utils.data import Dataset


class MedicalImageDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted(
            folder
            for folder in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, folder))
        )

        self.class_to_idx = {
            class_name: index
            for index, class_name in enumerate(self.classes)
        }

        self.images = []

        for class_name in self.classes:
            class_path = os.path.join(root_dir, class_name)

            for filename in os.listdir(class_path):
                if filename.lower().endswith(
                    (".jpg", ".jpeg", ".png", ".bmp", ".webp")
                ):
                    self.images.append(
                        (
                            os.path.join(class_path, filename),
                            self.class_to_idx[class_name]
                        )
                    )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path, label = self.images[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label