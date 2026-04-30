import cv2
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms


def apply_clahe(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def normalize(img):
    return img.astype(np.float32) / 255.0


class FingerJointDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df        = pd.read_csv(csv_file)
        self.img_dir   = img_dir
        self.transform = transform

        self.multitask = all(c in self.df.columns for c in ['JSN', 'OP', 'ER'])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        img_name = f"{row['id']}_{row['joint'].lower()}.png"
        img_path = f"{self.img_dir}/{img_name}"

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = apply_clahe(img)

        if self.transform:
            img = self.transform(img)

        label_kl = int(row['KL'])

        if self.multitask:
            labels = {
                'kl':  label_kl,
                'jsn': int(row['JSN']),
                'op':  int(row['OP']),
                'er':  int(row['ER']),
            }
        else:
            labels = {
                'kl':  label_kl,
                'jsn': -1,
                'op':  -1,
                'er':  -1,
            }

        return img, labels


def get_transforms(img_size=224,
                   mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]):
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    val_test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    return train_transform, val_test_transform


def get_dataloaders(img_size=224, batch_size=32,
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]):

    train_csv = 'train_multitask.csv'
    val_csv   = 'val_multitask.csv'
    test_csv  = 'test_multitask.csv'
    train_dir = 'finger_joints_img_train'
    val_dir   = 'finger_joints_img_val'
    test_dir  = 'finger_joints_img_test'

    train_transform, val_test_transform = get_transforms(img_size, mean, std)

    train_dataset = FingerJointDataset(train_csv, train_dir, transform=train_transform)
    val_dataset   = FingerJointDataset(val_csv,   val_dir,   transform=val_test_transform)
    test_dataset  = FingerJointDataset(test_csv,  test_dir,  transform=val_test_transform)

    labels         = [int(train_dataset.df.iloc[i]['KL']) for i in range(len(train_dataset))]
    class_counts   = Counter(labels)
    sample_weights = [1.0 / class_counts[l] for l in labels]
    sampler        = WeightedRandomSampler(sample_weights, len(labels), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,   num_workers=0)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False,   num_workers=0)

    return train_loader, val_loader, test_loader