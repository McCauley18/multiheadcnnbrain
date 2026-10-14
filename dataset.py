import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split

from config import (
    DATA_DIR, CLASSES, IMAGE_SIZE, BATCH_SIZE,
    NUM_WORKERS, RANDOM_SEED
)


class BrainTumorDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = CLASSES
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.images = []
        self.labels = []

        for cls in self.classes:
            class_dir = os.path.join(root_dir, cls)
            if not os.path.exists(class_dir):
                class_dir = os.path.join(root_dir, 'train', cls)
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(class_dir, img_name))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    return train_transform, eval_transform


def build_dataloaders(data_dir=DATA_DIR):
    train_transform, eval_transform = get_transforms()

    train_full = BrainTumorDataset(data_dir, transform=train_transform)
    eval_full = BrainTumorDataset(data_dir, transform=eval_transform)

    labels = train_full.labels
    indices = list(range(len(labels)))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=labels
    )

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=[labels[i] for i in train_val_idx]
    )

    train_dataset = Subset(train_full, train_idx)
    val_dataset = Subset(eval_full, val_idx)
    test_dataset = Subset(eval_full, test_idx)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True
    )

    return train_loader, val_loader, test_loader, train_full