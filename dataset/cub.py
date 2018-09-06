import os
from . import utils
import torch
import torchvision
import numpy as np
import PIL.Image
import tarfile


class Birds(torch.utils.data.Dataset):
    def __init__(self, root, labels, transform = None):

        # e.g., labels = range(0, 50) for using first 50 classes only
        self.labels = labels
        if transform: self.transform = transform
        # print("OKOKO", root)
        self.ys, self.im_paths = [], []
        for i in torchvision.datasets.ImageFolder(
            root = os.path.join(root, 'images')
        ).imgs:
            # i[1]: label, i[0]: path to file, including root
            y = i[1] 
            # fn needed for removing non-images starting with `._`
            fn = os.path.split(i[0])[1]
            if y in self.labels and fn[:2] != '._':
                self.ys  += [y]
                self.im_paths.append(i[0])

    def nb_classes(self):
        n = len(np.unique(self.ys))
        assert n == len(self.labels)
        return n

    def __len__(self):
        return len(self.ys)
        
    def __getitem__(self, index):
        im = PIL.Image.open(self.im_paths[index])
        im = self.transform(im)
        return im, self.ys[index]