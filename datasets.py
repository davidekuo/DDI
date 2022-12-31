import os
import pandas as pd
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
from torchvision.io import read_image

"""Constants and transforms copied from DDI-Code"""
means = [0.485, 0.456, 0.406]
stds = [0.229, 0.224, 0.225]
test_transform = T.Compose([
    lambda x: x.convert('RGB'),
    T.Resize(299),
    T.CenterCrop(299),
    T.ToTensor(),
    T.Normalize(mean=means, std=stds)
])


def params_to_list(params, defaults: list):
    """
    Helper function for DDI_DataModule class

    Args:
    - params (Any): parameters to convert into a list (may already be a list)
    - defaults (list): list of default parameters to return if params is None

    Returns:
    - list of parameters
    """
    if params is None:
        return defaults
    else:
        return params if isinstance(params, list) else [params]


class DDI_Dataset(Dataset):
    """
    PyTorch Dataset class for Diverse Dermatology Images dataset
    """

    def __init__(self, annotations_df, img_dir, transform=None):
        """
        Args:
          - annotations_df (Pandas DataFrame): DataFrame of annotations for dataset
          - img_dir (str): path to directory storing DDI images
          - transform (torchvision.transforms): torchvision transforms for dataset images
        """
        super().__init__()
        self.annotations = annotations_df
        self.img_dir = img_dir  # './ddi_data/'
        self.transform = transform

    def __len__(self):
        """Returns size of dataset"""
        return len(self.annotations)

    def __getitem__(self, idx):
        """
        For idx-th data point, returns:
          - image (Tensor): PyTorch tensor representation of DDI image
          - skin_tone (int): FST category [12, 34, 56]
          - malignant (bool): True if lesion is malignant, False if not
          - disease (string): diagnosis of lesion
          - disease_label (int): int label corresponding to disease
        """
        img_path = self.img_dir + os.sep + self.annotations.loc[idx, 'DDI_file']  # str
        image = read_image(img_path)  # Tensor
        skin_tone = self.annotations.loc[idx, 'skin_tone']  # int
        malignant = self.annotations.loc[idx, 'malignant']  # boolean
        disease = self.annotations.loc[idx, 'disease']  # str
        disease_label = self.annotations.loc[idx, 'disease_label']  # int

        if self.transform:
            image = self.transform(image)

        return image, skin_tone, malignant, disease, disease_label


class DDI_DataModule(pl.LightningDataModule):
    """LightningDataModule for Diverse Dermatology Images dataset"""

    def __init__(self,
                 random_seed,
                 batch_size,
                 num_workers,
                 metadata_csv,
                 img_dir,
                 transform=None,
                 skin_tone=None,
                 malignant=None,
                 diseases=None):
        """
        Args:
          - random_seed (int): random seed for generating train/val/test splits
          - batch_size (int): batch size for DataLoader
          - num_workers (int): num_workers for DataLoader
          - metadata_csv (str): path to metadata CSV file ('ddi_metadata.csv')
          - img_dir (str): path to directory storing DDI images
          - transform (torchvision.transforms): torchvision transforms for dataset images
          - skin_tone (list[int]): FST skin tone categories to include. None -> include all skin tones
          - malignant (list[bool]): malignancy categories to include. None -> include both malignant and benign
          - diseases (list[str]): diagnoses to include. None -> include all diagnoses
        """
        super().__init__()
        # parameters for DataLoader
        self.seed = random_seed
        self.batch_size = batch_size
        self.num_workers = num_workers

        # parameters for DDI_DataSet
        self.metadata = pd.read_csv(metadata_csv, index_col=0)  # './ddi_data/ddi_metadata.csv'
        self.img_dir = img_dir  # './ddi_data/'
        self.transform = transform

        # Create dictionary mapping disease (str) to label (int)
        self.disease2label = {disease: label for label, disease in enumerate(sorted(self.metadata['disease'].unique()))}
        # Add 'disease_label' col to metadata with int label for each diagnosis
        self.metadata['disease_label'] = self.metadata['disease'].map(self.disease2label)

        # Create DDI_Dataset of specified skin tones, malignancy status, and diagnoses
        # 1. parse inputs into lists for downstream processing
        skin_tone = params_to_list(skin_tone, [12, 34, 56])
        malignant = params_to_list(malignant, [True, False])
        diseases = params_to_list(diseases, list(self.disease2label))
        # 2. make sure specified inputs are valid
        for s in skin_tone:
            assert s in [12, 34, 56], f"{s} is not a valid skin tone"
        for d in diseases:
            assert d in self.disease2label, f"{d} is not a valid diagnosis"
        # 3. select rows of metadata DataFrame that meet specifications
        indices = np.where(self.metadata['skin_tone'].isin(skin_tone) &
                           self.metadata['malignant'].isin(malignant) &
                           self.metadata['disease'].isin(diseases))
        selected_metadata = self.metadata.loc[indices]
        # 4. create DDI_Dataset
        self.dataset = DDI_Dataset(selected_metadata, img_dir, transform)

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):  # 'fit', 'validate', 'test', 'predict'
        self.train_data, self.val_data, self.test_data = random_split(self.dataset,
                                                                      [0.6, 0.2, 0.2],
                                                                      generator=torch.Generator().manual_seed(
                                                                          self.seed))

    def train_dataloader(self):
        return DataLoader(self.train_data,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_data,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True,
                          shuffle=True)
