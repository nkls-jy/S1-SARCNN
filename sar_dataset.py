import os
import torch
import numpy as np
from torch.utils.data import Dataset
import rasterio

"""
filtering for images with image ending
"""
def image_filter(filename):
    from torchvision.datasets.folder import IMG_EXTENSIONS
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

"""
image loader 
"""
def img_loader(path):
    with rasterio.open(path) as f:
        img = f.read()
    return img

"""
finding files 
"""
def find_files(dir, filter=None):
    images = list()
    if filter is None:
        filter = lambda x: True

    for fname in sorted(os.listdir(dir)):
        if filter(fname):
            images.append(os.path.join(dir, fname))
    
    return images

"""
classes for dataset & dataloader
"""

class CenterCropNy(object):
    def __init__(self, size) -> None:
        self.size = size
    def __call__(self, img):
        y1 = (img.shape[1] - self.size) // 2
        x1 = (img.shape[2] - self.size) // 2
        y2 = y1 + self.size
        x2 = x1 + self.size
        return img[:, y1:y2, x1:x2]
    def __repr__(self):
        return self.__class__.__name__ + '(size{self.size})'

class Random8OrientationNy(object):
    def __init__(self):
        pass
    def __call__(self, img):
        k = img.shape[0]
        rot = np.random.randint(0, 8)

        # rotate along y and x axis
        img = np.rot90(img, axes=(1,2), k=rot)
        if rot > 3:
            img = img[:, ::-1, :]
        assert(img.shape[0] == k)
        return img

class RandomCropNy(object):
    def __init__(self, size):
        self.size = size
    def __call__(self, img):
        y1 = np.random.randint(0, img.shape[1] - self.size)
        x1 = np.random.randint(0, img.shape[2] - self.size)
        y2 = y1 + self.size
        x2 = x1 + self.size
        return img[:, y1:y2, x1:x2]
    def __repr__(self):
        return self.__class__.__name__ + '(size={0})'.format(self.size)

class NumpyToTensor(object):
    def __call__(self, x):
        return torch.from_numpy(x.copy(order='C'))


class PlainImageFolder(Dataset):
    def __init__(self, dirs, transform = None, cache = False, loader = img_loader, filter = image_filter):
        self.cache = cache
        self.img_cache = {}
        if isinstance(dirs, list):
            imgs = list()
            for r in dirs:
                imgs.extend(find_files(r, filter = filter))
        else:
            imgs = find_files(dirs, filter = filter)

        if len(imgs) == 0:
            raise(RuntimeError(f"Found 0 images in subfolder of: {dirs}"))

        self.dirs = dirs
        self.imgs = imgs
        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        path = self.imgs[index]
        if not index in self.img_cache:
            img = self.loader(path)
            if self.cache:
                self.img_cache[index] = img
        else:
            img = self.img_cache[index]
        
        if self.transform is not None:
            img = self.transform(img)

        return img
    
    def __len__(self):
        return len(self.imgs)

class PlainSarFolder(PlainImageFolder):
    def __init__(self, dirs, transform=None, cache=False):
        PlainImageFolder.__init__(self, dirs, transform=transform, cache=cache, loader=img_loader, filter=image_filter)
        


