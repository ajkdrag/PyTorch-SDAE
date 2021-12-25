import torch
from pathlib import Path
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms


class CustomDataset(Dataset):
    def __init__(self, root_dir, ext="jpg", transform=None):
        self.root_path = Path(root_dir)
        self.all_files = list(self.root_path.glob(f"*.{ext}"))
        self.transform = transform

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.item()
        img_path = self.all_files[index]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image


def get_transform(img_size):
    new_h, new_w = img_size
    return transforms.Compose(
        [transforms.Resize(size=(new_h, new_w)), transforms.ToTensor()]
    )


def get_dataloader(
    dataset_type,
    data_dir,
    batch_size,
    img_size,
    shuffle,
    val_split,
    num_workers,
    pin_memory,
    seed=42,
):

    transform = get_transform(img_size)
    if dataset_type.lower() == "mnist":
        dataset = datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
    else:
        dataset = CustomDataset(data_dir, transform=transform)

    total_val = int(val_split * len(dataset))
    total_train = len(dataset) - total_val
    train_dataset, val_dataset = random_split(
        dataset, (total_train, total_val), torch.Generator().manual_seed(seed)
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
