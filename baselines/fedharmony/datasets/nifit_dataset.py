# Nicola Dinsdale 2021
# Pytorch dataset for nifti files
########################################################################################################################
# Import dependencies
import torch
from torch.utils.data import Dataset
import numpy as np
import nibabel as nib
import random
import torchio as tio
from scipy.ndimage import zoom
import torch.nn.functional as F
########################################################################################################################
class nifti_dataset_ABIDE_agepred_domain(Dataset):
    def __init__(self, subjects, age_dict, site):
        self.subjects = subjects
        self.age_dict = age_dict
        self.site = site

    def __getitem__(self, index):
        subj = self.subjects[index]
        # print("id: "+str(subj.split('\\')[11]))
        pth = subj #+ '/mprage.anat/T1.nii.gz'
        data = nib.load(pth).get_fdata()[:, :, 5:165]
        # print("data shape:- "+str(data.shape))
        data = zoom(data, (128 / data.shape[0], 240 / data.shape[1], 160 / data.shape[2]))
        data = np.reshape(data, (1, 128, 240, 160))
        data = data / np.percentile(data, 99.9)
        data = data - np.mean(data)
        data = torch.from_numpy(data).float()

        parts = subj.split('\\')
        label = float(self.age_dict[parts[11]])

        if self.site == 'a':
            d = np.ones((1,)) * 0
        elif self.site == 'b':
            d = np.ones((1,)) * 1
        elif self.site == 'c':
            d = np.ones((1,)) * 2
        elif self.site == 'd':
            d = np.ones((1,)) * 3
        else:
            raise Exception('Unknown Site')
        d = torch.from_numpy(d)

        return data, label, d

    def __len__(self):
        return len(self.subjects)
