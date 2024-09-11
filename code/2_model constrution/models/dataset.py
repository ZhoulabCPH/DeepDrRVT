import os
import torchvision
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from torch import nn, optim
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
import pandas as pd
import numpy as np
import warnings
from augmentation import *

# Ignore warnings for cleaner output
warnings.filterwarnings('ignore')

# Set AMP (Automatic Mixed Precision) to True for potential performance improvement
is_amp = True

############################################################
# Folds
############################################################

def remove_last_if_8chars(s):
    """Removes the last character of a string if the string length is 8."""
    return s[:-1] if len(s) == 8 else s

def make_Bag_WSI(csvpath, clinicalpath, type):
    """
    This function loads the patch-level data from a CSV file and merges it 
    with clinical data based on SampleID. Adjustments are made based on the type of data.
    """
    # Load CSV based on the type
    if type != "CC":
        MyPandas = pd.read_csv(csvpath).iloc[:, 1:]
        MyPandas.rename(columns={'Patch_Name': 'Name'}, inplace=True)
    else:
        MyPandas = pd.read_csv(csvpath).iloc[:, 1:-1]
    # Merge clinical data with patch data
    MyPandas = pd.merge(MyPandas, Myclinicalpath, how='inner', on='SampleID')
    InputData = list(MyPandas.groupby(by="SampleID"))
    
    return InputData

def resample_if_needed(dataframes, number):
    """
    Resamples the dataframe if the number of rows is less than the required `number`. 
    If more rows are present, it randomly shuffles and selects `number` rows.
    """
    if dataframes.shape[0] < number:
        rows_to_resample = number - dataframes.shape[0]
        resampled_df = dataframes.sample(n=rows_to_resample, replace=True)
        df = pd.concat([dataframes, resampled_df], ignore_index=True)
    else:
        Index = np.arange(dataframes.shape[0])
        np.random.shuffle(Index)
        df = dataframes.iloc[Index[0:number]].reset_index(drop=True)

    return df

# Custom image transformation class for Gaussian Blur
class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

# Custom image transformation class for Solarization
class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img, 15)
        else:
            return img

# Double transformation (no specific transforms applied here, just a placeholder)
class Transform:
    def __init__(self):
        self.transform = transforms.Compose([])
        self.transform_prime = transforms.Compose([])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2

# Single transformation (Resize, ToTensor, Normalize)
class Transform_:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        return y1, y1

# Dataset class for validation
class HubmapDataset_Val(Dataset):
    def __init__(self, df, patchnumbers, arg):
        self.number = patchnumbers
        self.df = df
        self.arg = arg
        self.length = len(self.df)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        r = {}
        mydataframes_ = list(self.df)[index][1]
        Datasets = mydataframes_
        Name = list(self.df)[index][0]
        MaxFeature = int(self.arg.Feature_dim) + 1

        # Extract patch-level features and metadata
        Values = torch.from_numpy(np.array(Datasets.values[:, 1:MaxFeature], dtype=np.float32))
        Patch_Name = Datasets['Name'].values
        label = torch.from_numpy(np.array(Datasets['T_RVT'].values, dtype=np.float32))
        time_DFS = np.array(Datasets['time_DFS'].values, dtype=np.float32)[0]
        DFS = np.array(Datasets['DFS'].values, dtype=np.int32)[0]

        # Store in dictionary
        r['index'] = index
        r['values'] = Values
        r['label'] = label[0]
        r['time_DFS'] = torch.tensor(time_DFS)
        r['DFS'] = torch.tensor(DFS)
        r['Patient_Name'] = Name
        r['Patch_Name'] = Patch_Name

        return r

# Dataset class for training (with resampling)
class HubmapDataset_Train(Dataset):
    def __init__(self, df, patchnumbers, arg):
        self.number = patchnumbers
        self.df = df
        self.arg = arg
        self.length = len(self.df)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        r = {}
        mydataframes_ = resample_if_needed(list(self.df)[index][1], self.number)
        Datasets = mydataframes_
        Name = list(self.df)[index][0]
        MaxFeature = int(self.arg.Feature_dim) + 1

        # Extract patch-level features and metadata
        Values = torch.from_numpy(np.array(Datasets.values[:, 1:MaxFeature], dtype=np.float32))
        Patch_Name = Datasets['Name'].values
        label = torch.from_numpy(np.array(Datasets['T_RVT'].values, dtype=np.float32))
        time_DFS = np.array(Datasets['time_DFS'].values, dtype=np.float32)[0]
        DFS = np.array(Datasets['DFS'].values, dtype=np.int32)[0]

        # Store in dictionary
        r['index'] = index
        r['values'] = Values
        r['label'] = label[0]
        r['time_DFS'] = torch.tensor(time_DFS)
        r['DFS'] = torch.tensor(DFS)
        r['Patient_Name'] = Name
        r['Patch_Name'] = Patch_Name

        return r

# Image to tensor conversion (with optional mode)
def image_to_tensor(image, mode='bgr'):
    if mode == 'bgr':
        image = image[:, :, ::-1]  # Convert from BGR to RGB
    x = image
    x = x.transpose(2, 0, 1)
    x = np.ascontiguousarray(x)
    x = torch.tensor(x, dtype=torch.float)
    return x

# Tensor to image conversion (with optional mode)
def tensor_to_image(x, mode='bgr'):
    image = x.data.cpu().numpy()
    image = image.transpose(1, 2, 0)
    if mode == 'bgr':
        image = image[:, :, ::-1]  # Convert from RGB to BGR
    image = np.ascontiguousarray(image)
    image = image.astype(np.float32)
    return image

# List of tensor fields for stacking during batch collation
tensor_list = ['values', 'label', 'time_DFS', 'DFS']

# Custom collate function for DataLoader
def null_collate(batch):
    """
    Custom collate function for DataLoader to combine batch elements into a dictionary
    and stack tensors for certain fields.
    """
    d = {}
    key = batch[0].keys()
    for k in key:
        v = [b[k] for b in batch]
        if k in tensor_list:
            v = torch.stack(v)
        d[k] = v
    return d
