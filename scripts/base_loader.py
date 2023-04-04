import gc
import glob
import os
import warnings

import matplotlib.pyplot as plt
import cv2
import numpy as np

import albumentations as A
from torch import nn
from torch.utils.data import DataLoader, Dataset

class VCID_Dataset(Dataset):
    def __init__(self, imgs, surface_vols, CFG, mode="train", transform=None):
        self.imgs = imgs
        self.surface_vols = surface_vols
        self.img_size = CFG["img_size"]
        self.transform = transform
        self.mode = mode
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
            for grid_idx in grid_indices:
                grid_imgidx_list = [img_idx]
                grid_imgidx_list.extend(grid_idx)
                flatten_grid.append(grid_imgidx_list)
        self.flatten_grid = flatten_grid
        return self.flatten_grid

    # def read_surface_vols(self, img_idx, grid_idx):
    #     surface_vols = []
    #     for i in range(6):
    #         surface_path = os.path.join(CFG["INPUT_DIR"], self.mode, str(img_idx+1), "surface_volume", f"{i:02}.tif")
    #         surface_vol = cv2.imread(surface_path, cv2.IMREAD_GRAYSCALE)
    #         surface_vol = surface_vol[grid_idx[0]*self.img_size[0]:(grid_idx[0]+1)*self.img_size[0],
    #                                     grid_idx[1]*self.img_size[1]:(grid_idx[1]+1)*self.img_size[1]]
    #         surface_vols.append(surface_vol)
    #     return surface_vols

    def __len__(self):
        return len(self.flatten_grid)

    def __getitem__(self, idx):
        grid_idx = self.flatten_grid[idx]
        img_idx = grid_idx[0]
        grid_idx = grid_idx[1:]
        img = self.imgs[img_idx]
        surface_vol = self.surface_vols[img_idx]
        img = img[grid_idx[0]*self.img_size[0]:(grid_idx[0]+1)*self.img_size[0],
                    grid_idx[1]*self.img_size[1]:(grid_idx[1]+1)*self.img_size[1]]
        surface_vol_list = []
        for surface_slice in surface_vol:
            surface_slice = surface_slice[grid_idx[0]*self.img_size[0]:(grid_idx[0]+1)*self.img_size[0],
                                            grid_idx[1]*self.img_size[1]:(grid_idx[1]+1)*self.img_size[1]]
            surface_vol_list.append(surface_slice)
        img = np.stack([img] + surface_vol_list, axis=0)
        if self.transform:
            img = self.transform(image=img)["image"]
        return img

def read_surfacevol_all(img_idx, mode):
    surface_vols = []
    for i in range(10):
        print("\r", f"reading idx:{i+1}", end="")
        surface_path = os.path.join(CFG["INPUT_DIR"], mode, str(img_idx+1), "surface_volume", f"{i:02}.tif")
        surface_vol = cv2.imread(surface_path, cv2.IMREAD_GRAYSCALE)
        surface_vols.append(surface_vol)
    print(f"  => read surface volume done. [{img_idx+1}]")
    return surface_vols

if __name__=="__main__":
    BASE_DIR = "/working/"
    INPUT_DIR = os.path.join(BASE_DIR, "input")
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    CFG = {
        "img_size": [256, 256],
        "batch_size": 4,
        "INPUT_DIR": INPUT_DIR,
    }
    mask_1 = cv2.imread(os.path.join(TRAIN_DIR, "1", "mask.png"), cv2.IMREAD_GRAYSCALE)  
    mask_2 = cv2.imread(os.path.join(TRAIN_DIR, "2", "mask.png"), cv2.IMREAD_GRAYSCALE)
    mask_3 = cv2.imread(os.path.join(TRAIN_DIR, "3", "mask.png"), cv2.IMREAD_GRAYSCALE)
    imgs = [mask_1, mask_2, mask_3]
    print("read surface volume 1")
    surface_vols_1 = read_surfacevol_all(0, "train")
    print("read surface volume 2")
    surface_vols_2 = read_surfacevol_all(1, "train")
    print("read surface volume 3")
    surface_vols_3 = read_surfacevol_all(2, "train")
    print("dataset")
    surface_vols = [surface_vols_1, surface_vols_2, surface_vols_3]
    dataset = VCID_Dataset(imgs, surface_vols, CFG, mode="train")
    print("dataloader")
    dataloader = DataLoader(dataset, batch_size=CFG["batch_size"], shuffle=True, num_workers=2)
    
    print("check loader")
    for batch_idx, imgs in enumerate(dataloader):
        print("batch_idx: ", batch_idx)
        print(imgs.shape)
        if batch_idx >= 0:
            break