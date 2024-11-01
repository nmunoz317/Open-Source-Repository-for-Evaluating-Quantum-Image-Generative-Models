import os
import numpy as np
from numpy import load

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

import cv2
import nibabel as nib
from skimage import color



class BloodDataset(Dataset):
    """Pytorch dataloader for the Blood MedMNIST Dataset"""

    def __init__(self, path, num_imgs=3000, image_shape=(32,32), num_classes=8, transform=None):
        """
        Args:
            path (str): Path to the dataset (.npz file).
            transform (callable, optional): Optional transform to be applied to each sample.
            num_imgs (int, optional): Number of images used for training.
            image_shape (tuple, optional): Shape of the images (height, width).
            num_classes (int, optional): Number of classes to load from the dataset.
        """
        self.path = path
        self.transform = transform
        # Load dataset from .npz file
        self.datag = np.load(self.path + 'bloodmnist.npz')
        # Extract training images from the dataset 
        imgs = self.datag['train_images']
        # Extract training labels from the dataset
        labels = self.datag['train_labels']
        self.height, self.width = image_shape

        # Select the desired number of images and filter by the number of classes
        filtered_indices = np.where(labels < num_classes)[0]
        self.data = imgs[filtered_indices][:num_imgs]
        self.labels = labels[filtered_indices][:num_imgs]

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieve an image and its corresponding label by index."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Convert the image to grayscale and resize it
        image = color.rgb2gray(self.data[idx, :])
        image = cv2.resize(image, dsize=(self.height, self.width), interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32)
        image = torch.tensor(image).unsqueeze(0)  # Add channel dimension for grayscale image

        # Normalize the image
        normalize_transform = transforms.Normalize((0.5,), (0.5,))
        image = normalize_transform(image)

        # Get the label
        label = self.labels[idx]

        # Apply additional transformations if provided
        if self.transform:
            image = self.transform(image)

        # Return the processed image and its label
        return image, label
    
class BrainDataset(Dataset):
    """Pytorch dataloader for the BRATS Dataset"""
    def __init__(self, path, slice=(155//2), num_imgs=1500, image_shape=(64,64), num_classes=4, transform=None):
        """
        Args:
            path (str): Path to the folder with the NIfTI files.
            slice (int, optional): Specific slice to extract from 3D MRI images.
            num_imgs (int, optional): Number of images to load.
            image_shape (tuple, optional): Shape of the images (height, width).
            num_classes (int, optional): Number of MRI modalities to load from the dataset.
            transform (callable, optional): Optional transform to be applied to each sample.
        """
        self.path = path
        self.transform = transform
        names=os.listdir(path)
        # Filter files
        names=[file for file in names if file.endswith('.nii.gz') and not file.startswith('._')]
        names = [item for item in names for _ in range(num_classes)]
        self.height, self.width=image_shape
        self.names = names[:num_imgs]
        self.slice=slice


    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.names)

    def __getitem__(self, idx):
        """Retrieve a brain image and its corresponding label."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load the NIfTI image and get its data
        img = nib.load(self.path+self.names[idx]).get_fdata()
        name = self.names[idx]
        mritype=self.names[:(idx+1)].count(name)

        # Normalize the image
        max_=np.max(img[:,:,:,mritype-1])
        min_=np.min(img[:,:,:,mritype-1])
        label=mritype-1
        img_re=cv2.resize(img[20:210,20:210,self.slice,mritype-1], dsize=(self.height, self.width), interpolation=cv2.INTER_LINEAR)
        normalized_data = ((img_re - min_) / (max_ - min_))
        image = normalized_data.astype(np.float32)
        image_tensor = torch.tensor(image).unsqueeze(0)
        normalize_transform = transforms.Normalize((0.5,), (0.5,))
        sample = normalize_transform(image_tensor)

        # Apply additional transformations if provided
        if self.transform:
            image = self.transform(image)

        # Return the processed image and its label
        return sample, label