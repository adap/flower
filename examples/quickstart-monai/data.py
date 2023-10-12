import os
import numpy as np
from monai.transforms import (
    EnsureChannelFirst,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    ToTensor,
)

from monai.data import Dataset, DataLoader
from urllib import request
import tarfile


def load_data():
    image_file_list, image_label_list, num_total, num_class = _download_data()
    trainX, trainY, valX, valY, testX, testY = _split_data(image_file_list, image_label_list, num_total)
    train_transforms, val_transforms = _get_transforms()

    train_ds = MedNISTDataset(trainX, trainY, train_transforms)
    train_loader = DataLoader(train_ds, batch_size=300, shuffle=True, num_workers=2)

    val_ds = MedNISTDataset(valX, valY, val_transforms)
    val_loader = DataLoader(val_ds, batch_size=300, num_workers=2)

    test_ds = MedNISTDataset(testX, testY, val_transforms)
    test_loader = DataLoader(test_ds, batch_size=300, num_workers=2)

    return train_loader, val_loader, test_loader, num_class


class MedNISTDataset(Dataset):

    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


def _download_data():
    data_dir = './MedNIST/'
    _download_and_extract("https://dl.dropboxusercontent.com/s/5wwskxctvcxiuea/MedNIST.tar.gz", os.path.join(data_dir))

    class_names = sorted([x for x in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, x))])
    num_class = len(class_names)
    image_files = [[os.path.join(data_dir, class_name, x) 
                    for x in os.listdir(os.path.join(data_dir, class_name))] 
                   for class_name in class_names]
    image_file_list = []
    image_label_list = []
    for i, class_name in enumerate(class_names):
        image_file_list.extend(image_files[i])
        image_label_list.extend([i] * len(image_files[i]))
    num_total = len(image_label_list)
    return image_file_list, image_label_list, num_total, num_class


def _split_data(image_file_list, image_label_list, num_total):
    valid_frac, test_frac = 0.1, 0.1
    trainX, trainY = [], []
    valX, valY = [], []
    testX, testY = [], []

    for i in range(num_total):
        rann = np.random.random()
        if rann < valid_frac:
            valX.append(image_file_list[i])
            valY.append(image_label_list[i])
        elif rann < test_frac + valid_frac:
            testX.append(image_file_list[i])
            testY.append(image_label_list[i])
        else:
            trainX.append(image_file_list[i])
            trainY.append(image_label_list[i])


    return trainX, trainY, valX, valY, testX, testY

def _get_transforms():
    train_transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        RandRotate(range_x=15, prob=0.5, keep_size=True),
        RandFlip(spatial_axis=0, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
        ToTensor()
    ])

    val_transforms = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensity(),
        ToTensor()
    ])
    
    return train_transforms, val_transforms

def _download_and_extract(url, dest_folder):
    if not os.path.isdir(dest_folder):
        # Download the tar.gz file
        tar_gz_filename = url.split("/")[-1]
        if not os.path.isfile(tar_gz_filename ):
            with request.urlopen(url) as response, open(tar_gz_filename, 'wb') as out_file:
                out_file.write(response.read())
        
        # Extract the tar.gz file
        with tarfile.open(tar_gz_filename, 'r:gz') as tar_ref:
            tar_ref.extractall()

