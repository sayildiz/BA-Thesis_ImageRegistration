import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class HistologyDataset(Dataset):
    """
    Histology dataset
    per Default: HE=Fixed_Image, PHH3 = Moving_Image

    Parameters:
        csv_file : str
            csv file with first_column: he_image and second_column: phh3_image
        swap: bool
            if true, swaps HE to Moving_Image, PHH3 to Fixed_Image

    Returns:
        [moving, fixed], [ground_truth]
    """
    def __init__(self, csv_file, transform_he=None, transform_phh3=None, swap=False):
        self.img_frame = pd.read_csv(csv_file)
        self.transform_he = transform_he
        self.transform_phh3 = transform_phh3
        self.swap = swap

    def __len__(self):
        return len(self.img_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        he_path = self.img_frame.iloc[idx, 0]
        phh3_path = self.img_frame.iloc[idx, 1]

        he_image = Image.open(he_path).convert('RGB')
        phh3_image = Image.open(phh3_path).convert('RGB')

        if self.transform_he:
            he_image = self.transform_he(he_image)
        if self.transform_phh3:
            phh3_image = self.transform_phh3(phh3_image)

        if self.swap:
            return [he_image, phh3_image], [phh3_image]
        else:
            return [phh3_image, he_image], [he_image]
