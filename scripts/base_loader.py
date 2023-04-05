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
    def __init__(self, CFG, mode="train", transform=None):
        # get config
        self.mode = mode
        if self.mode=="train":
            self.DATADIR = CFG["TRAIN_DIR"]
            self.data_idx_list = CFG["TRAIN_IDX_LIST"]
        elif self.mode=="valid":
            self.DATADIR = CFG["TRAIN_DIR"]
            self.data_idx_list = CFG["VALID_IDX_LIST"]
        elif self.mode == "test":
            self.DATADIR = CFG["TEST_DIR"]
            self.data_idx_list = CFG["TEST_IDX_LIST"]
        self.surface_num = CFG["surface_num"]
        self.img_size = CFG["img_size"]
        self.transform = transform
        # get imgs
        print("initializing dataset...")
        self.imgs = []
        for idx in self.data_idx_list:
            img_path = os.path.join(self.DATADIR, idx, "mask.png")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            # img = cv2.imread(img_path, 0)
            img = img.reshape(img.shape[0], img.shape[1], 1) # (h, w, channel=1)
            # img = np.stack(img, axis=0) # (h, w, channel=1)
            # print(img.shape)
            # if self.imgs is None:
            #     self.imgs = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) # (img_idx=1, h, w, channel=1)
            #     # self.imgs = np.stack(img, axis=0) # (img_idx=1, h, w, channel=1)
            #     print(self.imgs.shape, img.shape)
            # else:
            #     img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2]) # (img_idx=1, h, w, channel=1)
            #     # img = np.stack(img, axis=0) # (img_idx=1, h, w, channel=1)
            #     self.imgs = np.concatenate([self.imgs, img], axis=0) # (img_idx=N, h, w, channel=1)
            #     print(self.imgs.shape, img.shape)
            self.imgs.append(img)
        # self.imgs = np.stack(self.imgs, axis=2)
        # self.imgs = [cv2.imread(os.path.join(self.DATADIR, idx, "mask.png"), cv2.IMREAD_GRAYSCALE) for idx in self.data_idx_list]
        for img in self.imgs:
            assert img is not None, "img is None. data path is wrong"
        print("read imgs done.", np.array(self.imgs).shape)
        # get and split surface
        self.surface_vols = self.read_surfacevols()
        for surface_vol in self.surface_vols:
            assert surface_vol is not None, "surface_vol is None. data path is wrong"
        print("read surface volume done.", np.array(self.surface_vols).shape)
        # split grid
        self.get_all_grid()
        self.fileter_grid()
        self.get_flatten_grid()
        print("split grid done.") 
        # get label imgs
        if self.mode == "train" or self.mode == "valid":
            self.labels = []
            for idx in self.data_idx_list:
                label_path = os.path.join(self.DATADIR, idx, "inklabels.png")
                assert os.path.exists(label_path), f"{label_path} is not exist."
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                # label = np.stack(label, axis=2)# (h, w, cahnnel=1)
                label = label.reshape(label.shape[0], label.shape[1], 1) # (h, w, channel=1)
                # if self.labels is None:
                #     self.labels = np.stack(label, axis=0) # (img_idx=1, h, w, channel=1)
                # else:
                #     label = np.stack(label, axis=0) # (img_idx=1, h, w, channel=1)
                #     self.labels = np.concatenate([self.labels, label], axis=0) # (img_idx=N, h, w, channel=1)
                self.labels.append(label)# 画像サイズがそれぞれ違うので単純にconcatできずlist化しているs
        print("initializing dataset done.")

    def read_surfacevols(self):
        surface_vols = []
        print("reading surface volume...")
        for img_idx in self.data_idx_list:
            surface_vol_ = None
            for i in range(self.surface_num):
                print("\r", f"reading idx : {i+1}/{self.surface_num}", end="")
                surface_path = os.path.join(self.DATADIR, img_idx, "surface_volume", f"{i:02}.tif")
                surface_vol = cv2.imread(surface_path, cv2.IMREAD_GRAYSCALE)
                # surface_vol = np.stack(surface_vol, axis=2) # (h, w, channel=1)
                surface_vol = surface_vol.reshape(surface_vol.shape[0], surface_vol.shape[1], 1) # (h, w, channel=1)
                if surface_vol_ is None:
                    surface_vol_ = surface_vol
                else:
                    surface_vol_ = np.concatenate([surface_vol_, surface_vol], axis=2) # (h, w, channel=surface_num)
            # if surface_vols is None:
            #     surface_vols = np.stack(surface_vol_, axis=0) # (img_idx=1, h, w, channel=surface_num)
            # else:
            #     surface_vol_ = np.stack(surface_vol_, axis=0) # (img_idx=1, h, w, channel=surface_num)
            #     surface_vols = np.concatenate([surface_vols, surface_vol_], axis=0)# (img_idx=N, h, w, channel=surface_num)
            surface_vols.append(surface_vol_)
            print(f"  => read surface volume done. [{img_idx}]")
        return surface_vols


    def get_grid_img(self, img, grid_idx):
        img_grid = img[grid_idx[0]*self.img_size[0]:(grid_idx[0]+1)*self.img_size[0],
                        grid_idx[1]*self.img_size[1]:(grid_idx[1]+1)*self.img_size[1]]
        return img_grid 

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
                img_grid = self.get_grid_img(img, grid_idx)
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

    def __len__(self):
        return len(self.flatten_grid)

    def __getitem__(self, idx):
        grid_idx = self.flatten_grid[idx]
        img_idx = grid_idx[0]
        grid_idx = grid_idx[1:]
        img = self.imgs[img_idx]
        print("img shape : ", img.shape)
        surface_vol = self.surface_vols[img_idx]
        print("surface_vol shape : ", np.array(surface_vol).shape)
        img = self.get_grid_img(img, grid_idx)
        surface_vol = self.get_grid_img(surface_vol, grid_idx)
        print("surface_vol shape : ", surface_vol.shape)
        assert surface_vol.shape[0]==img.shape[0] and surface_vol.shape[1]==img.shape[1] , "surface_vol_list shape is not same as img shape"
        # surface_vol_list = []
        # for surface_channel_idx in range(surface_vol.shape[2]):
        #     surface_slice = surface_vol[:, :, surface_channel_idx]
        #     print("surface_slice shape : ", surface_slice.shape)
        #     surface_slice = self.get_grid_img(surface_slice, grid_idx)
        #     assert surface_slice.shape == img.shape, "surface_slice shape is not same as img shape"
        #     surface_vol_list.append(surface_slice)
        img = np.stack([img] + surface_vol, axis=0)
        
        if self.mode == "test":
            if self.transform:
                img = self.transform(image=img)["image"]
            return img
        elif self.mode == "train" or self.mode=="valid":
            label = self.labels[img_idx]
            print("label shape : ", np.array(label).shape)
            label = self.get_grid_img(label, grid_idx)
            if self.transform:
                transformed = self.transform(image=img, mask=label)
                img = transformed["image"]
                label = transformed["mask"]
            return img, label

if __name__=="__main__":
    BASE_DIR = "/working/"
    INPUT_DIR = os.path.join(BASE_DIR, "input")
    TRAIN_DIR = os.path.join(INPUT_DIR, "train")
    TEST_DIR = os.path.join(INPUT_DIR, "test")
    CFG = {
        "img_size": [256, 256],
        "batch_size": 4,
        "INPUT_DIR": INPUT_DIR,
        "TRAIN_DIR": TRAIN_DIR,
        "TEST_DIR": TEST_DIR,
        "surface_num": 3,
        "TRAIN_IDX_LIST": ["1", "2"],
        # "TRAIN_IDX_LIST" : ["1"],
        "VALID_IDX_LIST": ["3"],
        "TEST_IDX_LIST": ["a", "b"],
    }
    # mask_1 = cv2.imread(os.path.join(TRAIN_DIR, "1", "mask.png"), cv2.IMREAD_GRAYSCALE)  
    # mask_2 = cv2.imread(os.path.join(TRAIN_DIR, "2", "mask.png"), cv2.IMREAD_GRAYSCALE)
    # mask_3 = cv2.imread(os.path.join(TRAIN_DIR, "3", "mask.png"), cv2.IMREAD_GRAYSCALE)
    # imgs = [mask_1, mask_2, mask_3]
    # print("read surface volume 1")
    # surface_vols_1 = read_surfacevol_all(0, "train")
    # print("read surface volume 2")
    # surface_vols_2 = read_surfacevol_all(1, "train")
    # print("read surface volume 3")
    # surface_vols_3 = read_surfacevol_all(2, "train")
    # surface_vols = [surface_vols_1, surface_vols_2, surface_vols_3]
    
    print("dataset")
    dataset = VCID_Dataset(CFG, mode="train")
    print("dataloader")
    dataloader = DataLoader(dataset, batch_size=CFG["batch_size"], shuffle=True, num_workers=0)
    
    print("check loader")
    for batch_idx, imgs in enumerate(dataloader):
        print("batch_idx: ", batch_idx)
        print(np.array(imgs).shape)
        break