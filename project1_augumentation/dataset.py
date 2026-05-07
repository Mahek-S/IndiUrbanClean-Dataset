import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import os

class UrbanDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.label_map = {
            "clean_street": 0,
            "open_waste": 1,
            "construction_waste": 2,
            "dumpyard": 3,
            "overfilled_bins": 4
        }

        all_data = pd.read_csv(csv_file)

        # skip rows where the image file doesn't exist on disk
        mask = all_data.apply(
            lambda row: os.path.exists(
                os.path.join(root_dir, row['label'], row['filename'])
            ), axis=1
        )
        skipped = (~mask).sum()
        if skipped > 0:
            print(f"[dataset] Skipping {skipped} missing files from {csv_file}")

        self.data = all_data[mask].reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.root_dir, row['label'], row['filename'])
        image = Image.open(img_path).convert("RGB")
        label = self.label_map[row['label']]

        if self.transform:
            image = self.transform(image)

        return image, label, img_path
