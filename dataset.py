import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import numpy as np

class CarvanaDS(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", "_mask.gif"))

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        mask[mask == 255.0] = 1.0 #Assigning 1.0 as class label for the car

        if self.transform is not None:
            augmentations = self.transform(image= image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        return image, mask


def TVT_split(image_dir, mask_dir, split_ratio= [0.8, 0.1, 0.1], transform=None, batch_size=32,  num_workers=2, shuffle=True, pin_memory=True):
    dataset = CarvanaDS(image_dir, mask_dir, transform)

    #Splits the Dataset
    train_ds, val_ds, test_ds = torch.utils.data.random_split(dataset, split_ratio)

    #Creates relevant loaders
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory)
    val_loader = DataLoader(dataset=val_ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory)
    test_loader = DataLoader(dataset=test_ds, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle, pin_memory=pin_memory)


    out = {"train_ds": train_ds, "train_loader": train_loader,
           "val_ds": val_ds, "val_loader": val_loader,
           "test_ds": test_ds, "test_loader": test_loader}
    return out