import gc
import glob
import os
import warnings

import matplotlib.pyplot as plt
import cv2

import albumentations as A
from torch import nn
from torch.utils.data import DataLoader, Dataset

class VCID_Dataset(Dataset):
    def __init__(self, imgs, img_size, transform=None):
        self.imgs = imgs
        self.img_size = img_size
        self.transform = transform
        self.get_all_grid()
        self.fileter_grid()
        self.get_flatten_grid()

    def get_all_grid(self):
        self.grid_indices = []
        for img in self.imgs:
            self.x_grid_size = img.shape[0] // self.img_size[0]
            self.y_grid_size = img.shape[1] // self.img_size[1]
            grid_img = []
            for i in range(self.x_grid_size):
                for j in range(self.y_grid_size):
                    grid_img.append([i, j])
            self.grid_indices.append(grid_img)
        return self.grid_indices
          
    def fileter_grid(self):
        grid_indices_all = []
        for img, grid_indices in zip(self.imgs, self.grid_indices):
            grid_indices_copy = grid_indices.copy()
            for grid_idx in grid_indices:
                img_grid = img[grid_idx[0]*self.img_size[0]:(grid_idx[0]+1)*self.img_size[0],
                                grid_idx[1]*self.img_size[1]:(grid_idx[1]+1)*self.img_size[1]]
                if img_grid.sum() == 0:
                    grid_indices_copy.remove(grid_idx)
            grid_indices_all.append(grid_indices_copy)
        self.grid_indices = grid_indices_all
        return self.grid_indices

    def get_flatten_grid(self):
        flatten_grid = []
        for img_idx, grid_indices in enumerate(self.grid_indices):
            print(img_idx, grid_indices)
            for grid_idx in grid_indices:
                grid_imgidx_list = [img_idx]
                grid_imgidx_list.extend(grid_idx)
                flatten_grid.append(grid_imgidx_list)
        self.flatten_grid = flatten_grid
        return self.flatten_grid

    def __len__(self):
        return len(self.flatten_grid)

    def __getitem__(self, idx):
        grid_idx = self.flatten_grid[idx]
        img_idx = grid_idx[0]
        grid_idx = grid_idx[1:]
        img = self.imgs[img_idx]
        img = img[grid_idx[0]*self.img_size[0]:(grid_idx[0]+1)*self.img_size[0],
                    grid_idx[1]*self.img_size[1]:(grid_idx[1]+1)*self.img_size[1]]
        
        if self.transform:
            img = self.transform(image=img)["image"]
        
        
        return img

if __name__=="__main__":
    import psutil
    BASE_DIR = "/working/"
    INPUT_DIR = os.path.join(BASE_DIR, "input")
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    mem_before = psutil.virtual_memory() 
    print(mem_before.used)
    mask_1 = cv2.imread(os.path.join(TRAIN_DIR, "1", "mask.png"))  
    mask_2 = cv2.imread(os.path.join(TRAIN_DIR, "2", "mask.png"))
    mask_3 = cv2.imread(os.path.join(TRAIN_DIR, "3", "mask.png"))
    mem_after = psutil.virtual_memory() 
    print(mem_after.used)
    print(mem_after.used - mem_before.used)
    imgs = [mask_1, mask_2, mask_3]
    dataset = VCID_Dataset(imgs, (256, 256))