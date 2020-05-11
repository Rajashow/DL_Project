import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader


class SketchDataSet(Dataset):
    """ Sketch datset """

    def __init__(self, directory, is_train):
        # Is this testing or training data?
        if is_train:
            self.is_train = True
        else:
            self.is_train = False

        self.num_class = 0
        self.num_images = 0
        self.fnames = [None]*18
        self.fsize = [None]*18
        self.fimgsize = [None]*18

        i = 0
        for filename in os.listdir(directory):
            if filename.endswith(".npy"):
                self.num_class += 1
                data = np.load(directory + filename)
                self.fnames[i] = directory + filename
                if is_train:
                    self.num_images += int(len(data)*0.1)
                    self.fsize[i] = int(len(data)*0.1)
                    self.fimgsize[i] = self.num_images
                else:
                    self.num_images += int(len(data)*0.025)
                    self.fsize[i] = int(len(data)*0.025)
                    self.fimgsize[i] = self.num_images
                i += 1

    def size_of_class(self, ind):
        return self.fsize[ind]

    def num_of_classes(self):
        return self.num_class

    def __len__(self):
        return self.num_images

    def __getitem__(self, ind):
        if self.is_train:
            for i in range(0, self.num_class):
                if ind - self.fimgsize[i] < 0 and i == 0:
                    img_dat = np.load(self.fnames[0])[ind].reshape(
                        1, 28, 28).astype(np.float32)
                    return torch.from_numpy(img_dat), i
                elif ind - self.fimgsize[i] < 0:
                    img_dat = np.load(self.fnames[i])[
                        ind - self.fimgsize[i-1]].reshape(1, 28, 28).astype(np.float32)
                    return torch.from_numpy(img_dat), i
        else:
            for i in range(0, self.num_class):
                if ind - self.fimgsize[i] < 0 and i == 0:
                    np_dat = np.load(self.fnames[0])
                    ind_offset = int(len(np_dat)*0.1)

                    img_dat = np_dat[ind +
                                     ind_offset].reshape(1, 28, 28).astype(np.float32)
                    return torch.from_numpy(img_dat), i
                elif ind - self.fimgsize[i] < 0:
                    np_dat = np.load(self.fnames[0])
                    ind_offset = int(len(np_dat)*0.1)

                    img_dat = np_dat[ind + ind_offset - self.fimgsize[i-1]
                                     ].reshape(1, 28, 28).astype(np.float32)
                    return torch.from_numpy(img_dat), i
        return None, None
