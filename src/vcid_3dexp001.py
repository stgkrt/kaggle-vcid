import gc
import os
import random
import time
import math
import yaml

import cv2
import numpy as np
import matplotlib.pyplot as plt

# model
import torch
import torchvision
import torch.nn as nn
import timm
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision.transforms.functional as TF
import segmentation_models_pytorch as smp
from functools import partial

# data loader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

# training
from torch.optim import SGD, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, ReduceLROnPlateau, ExponentialLR, CyclicLR

# metric
from sklearn.metrics import fbeta_score, roc_auc_score
import wandb
import warnings
warnings.filterwarnings('ignore')


"""
Configureations
"""
DEBUG = True
EXP_NAME = "exp_3d001"
EXP_YAML_PAHT = os.path.join("/working", "output", EXP_NAME, "Config.yaml")
# read yaml file to CFG
with open(EXP_YAML_PAHT) as yaml_file:
    CFG = yaml.load(yaml_file, Loader=yaml.FullLoader)
os.makedirs(os.path.join("/working", "output", EXP_NAME, "imgs"), exist_ok=True)

CFG["EXP_NAME"] = EXP_NAME
CFG["DEBUG"] = DEBUG
CFG["OUTPUT_DIR"] = os.path.join("/working", "output", EXP_NAME)
# CFG["SUMMARY"] = f"{EXP_NAME}: PSPnet model:resnet152, grid img size 512, img size 512"
CFG["SUMMARY"] = f"{EXP_NAME}: efficientnet-base-mymodel, grid img size 256, img size 512,add layers, loss change"


if DEBUG:
    CFG["n_epoch"] = 1
    CFG["folds"] = [0]
    CFG["slide_pos_list"] = [[0,0]]


def init_logger(log_file=os.path.join(CFG["OUTPUT_DIR"], 'train.log')):
    """Output Log."""
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger
LOGGER = init_logger()



"""
General Utils
"""
def seed_everything(seed=42):
    #os.environ['PYTHONSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic =True
    torch.backends.cudnn.benchmark = False
seed_everything(CFG["random_seed"])

# device optimization
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
LOGGER.info(f'Using device: {device}')

def asMinutes(s):
    """Convert Seconds to Minutes."""
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    """Accessing and Converting Time Data."""
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))

class AverageMeter(object):
    """Computes and stores the average and current value."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def logging_metrics_epoch(CFG, fold, epoch, slice_idx,train_loss_avg, valid_loss_avg, score, threshold, auc_score):
    wandb.log({f"train/fold{fold}": train_loss_avg,
                f"valid/fold{fold}": valid_loss_avg,
                f"score/fold{fold}":score,
                f"score threshold/fold{fold}":threshold,
                f"auc/fold{fold}":auc_score,
                f"epoch/fold{fold}":epoch+slice_idx*CFG["n_epoch"],
                })


"""
SCORE UTILS
"""
def fbeta_numpy(targets, preds, beta=0.5, smooth=1e-5):
    y_true_count = targets.sum()
    ctp = preds[targets==1].sum()
    cfp = preds[targets==0].sum()
    beta_squared = beta * beta

    c_precision = ctp / (ctp + cfp + smooth)
    c_recall = ctp / (y_true_count + smooth)
    dice = (1 + beta_squared) * (c_precision * c_recall) / (beta_squared * c_precision + c_recall + smooth)

    return dice

def calc_fbeta_auc(mask, mask_pred):
    mask = mask.astype(int).flatten()
    mask_pred = mask_pred.flatten()

    best_th = 0
    best_dice = 0
    dice_list = [] 
    # for th in np.array(range(10, 50+1, 5)) / 100:
    for th in np.array(range(10, 100+1, 5)) / 100:
        # dice = fbeta_score(mask, (mask_pred >= th).astype(int), beta=0.5)
        dice = fbeta_numpy(mask, (mask_pred >= th).astype(int), beta=0.5)
        dice_list.append(dice)
        # print(f'\t th: {th}, fbeta: {dice}')
        if dice > best_dice:
            best_dice = dice
            best_th = th
    
    auc = roc_auc_score(mask, mask_pred)
    # Logger.info(f'best_th: {best_th}, fbeta: {best_dice}')
    return best_dice, best_th, auc, dice_list


def calc_cv(mask_gt, mask_pred):
    best_dice, best_th, auc, dice_list = calc_fbeta_auc(mask_gt, mask_pred)
    return best_dice, best_th, auc, dice_list


"""
MODEL
"""
def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 conv1_t_size=7,
                 conv1_t_stride=1,
                 shortcut_type='B',
                 widen_factor=1.0):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(conv1_t_size, 7, 7),
                               stride=(conv1_t_stride, 2, 2),
                               padding=(conv1_t_size // 2, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        self.layer1 = self._make_layer(block,
                                       block_inplanes[0],
                                       layers[0],
                                       shortcut_type,
                                       stride=(1, 1, 1),
                                       downsample=False)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=(1, 2, 2),
                                       downsample=True)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=(1, 2, 2),
                                       downsample=True)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=(1, 2, 2),
                                       downsample=True)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride, downsample):
        downsample_block = None
        if downsample:
            if shortcut_type == 'A':
                downsample_block = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample_block = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample_block))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return [x1, x2, x3, x4]


def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]
    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)
    return model

class Decoder(nn.Module):
    def __init__(self, encoder_dims, upscale):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(encoder_dims[i]+encoder_dims[i-1], encoder_dims[i-1], 3, 1, 1, bias=False),
                nn.BatchNorm2d(encoder_dims[i-1]),
                nn.ReLU(inplace=True)
            ) for i in range(1, len(encoder_dims))])

        self.logit = nn.Conv2d(encoder_dims[0], 1, 1, 1, 0)
        self.up = nn.Upsample(scale_factor=upscale, mode="bilinear")

    def forward(self, feature_maps):
        for i in range(len(feature_maps)-1, 0, -1):
            f_up = F.interpolate(feature_maps[i], scale_factor=2, mode="bilinear")
            f = torch.cat([feature_maps[i-1], f_up], dim=1)
            f_down = self.convs[i-1](f)
            feature_maps[i-1] = f_down

        x = self.logit(feature_maps[0])
        mask = self.up(x)
        return mask


class SegModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = generate_model(model_depth=18, n_input_channels=1)
        self.decoder = Decoder(encoder_dims=[64, 128, 256, 512], upscale=4)
        
    def forward(self, x):
        feat_maps = self.encoder(x)
        feat_maps_pooled = [torch.mean(f, dim=2) for f in feat_maps]
        pred_mask = self.decoder(feat_maps_pooled)
        return pred_mask
    
    def load_pretrained_weights(self, state_dict):
        # Convert 3 channel weights to single channel
        # ref - https://timm.fast.ai/models#Case-1:-When-the-number-of-input-channels-is-1
        conv1_weight = state_dict['conv1.weight']
        state_dict['conv1.weight'] = conv1_weight.sum(dim=1, keepdim=True)
        print(self.encoder.load_state_dict(state_dict, strict=False))

model = SegModel()
x = torch.randn(1, 1, len(CFG["SURFACE_LIST"][0]), 512, 512)
print(x.shape)
out = model(x)
print(out.shape)

raise Exception("stop")

""" 
transfomrs
"""
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.2),
    A.VerticalFlip(p=0.2),
    A.RandomRotate90(p=0.2),
    A.RandomCrop(int(CFG["img_size"][0]*0.8), int(CFG["img_size"][1]*0.8), p=0.2),
    A.Blur(blur_limit=3, p=0.1),
    A.Resize(CFG["input_img_size"][0], CFG["input_img_size"][1]),
    ToTensorV2(),
])

valid_transforms = A.Compose([
    A.Resize(CFG["input_img_size"][0], CFG["input_img_size"][1]),
    ToTensorV2(),
])

"""
Dataset
"""
class VCID_Dataset(Dataset):
    def __init__(self, CFG, data_dir_list, surface_list, surface_volumes=None, slide_pos=[0,0], mode="train", transform=None):
        # get config
        self.mode = mode
        self.RANDOM_SLIDE = CFG["RANDOM_SLIDE"]
        self.img_size = CFG["img_size"]
        if self.mode=="train":  self.DATADIR = CFG["TRAIN_DIR"]
        elif self.mode=="valid":    self.DATADIR = CFG["TRAIN_DIR"]
        elif self.mode == "test":   self.DATADIR = CFG["TEST_DIR"]
        self.data_dir_list = data_dir_list
        self.surface_list = surface_list
        self.slide_pos = slide_pos
        self.transform = transform
        # get imgs
        # print("initializing dataset...")
        self.imgs = []
        for data_dir in self.data_dir_list:
            img_path = os.path.join(self.DATADIR, data_dir, "mask.png")
            # print(img_path)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = img.reshape(img.shape[0], img.shape[1], 1) # (h, w, channel=1)
            assert img is not None, "img is None. data path is wrong"
            self.imgs.append(img)  
        
        # get and split surface
        if surface_volumes is None:
            self.surface_vols = self.read_surfacevols()
        else:
            # print("using loaded surface_vols")
            self.surface_vols = surface_volumes
       
        # split grid
        self.get_all_grid()
        self.fileter_grid()
        self.get_flatten_grid()
        # print("split grid done.") 
       
        # get label imgs
        if self.mode == "train" or self.mode == "valid":
            self.labels = []
            for data_dir in self.data_dir_list:
                label_path = os.path.join(self.DATADIR, data_dir, "inklabels.png")
                assert os.path.exists(label_path), f"{label_path} is not exist."
                # read label
                label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                label = label.reshape(label.shape[0], label.shape[1], 1) # (h, w, channel=1)
                self.labels.append(label)# 画像サイズがそれぞれ違うので単純にconcatできずlist化しているs
        # print("initializing dataset done.")

    def get_surface_volumes(self):
        return self.surface_vols

    def read_surfacevols(self):
        """ read surface volume by data_dir_list and surface_list 
            Returns:surface_vuls (list): surface volume list [array(h,w,channel=surface_num), array(), ...]
        """
        surface_vols = []
        # print("reading surface volume...")
        for data_dir in self.data_dir_list:
            surface_vol_ = None
            for read_idx, surface_idx in enumerate(self.surface_list):
                # print("\r", f"reading idx : {read_idx+1}/{len(self.surface_list)}", end="")
                surface_path = os.path.join(self.DATADIR, data_dir, "surface_volume", f"{surface_idx:02}.tif")
                surface_vol = cv2.imread(surface_path, cv2.IMREAD_GRAYSCALE)
                surface_vol = surface_vol.reshape(surface_vol.shape[0], surface_vol.shape[1], 1) # (h, w, channel=1)
                if surface_vol_ is None:
                    surface_vol_ = surface_vol
                else:
                    surface_vol_ = np.concatenate([surface_vol_, surface_vol], axis=2) # (h, w, channel=surface_num)
            surface_vols.append(surface_vol_)
            # print(f"  => read surface volume done. [{data_dir}]")
        return surface_vols


    def get_grid_img(self, img, grid_idx):
        """ crop grid img from original img"""
        img_grid = img[(grid_idx[0]*self.img_size[0]) + self.slide_pos[0] : ((grid_idx[0]+1)*self.img_size[0]) + self.slide_pos[0],
                        (grid_idx[1]*self.img_size[1]) + self.slide_pos[1] : ((grid_idx[1]+1)*self.img_size[1]) + self.slide_pos[1]]
        return img_grid
    
    def get_grid_img_and_mask(self, img, mask, grid_idx):
        """ crop grid img from original img"""
        if self.RANDOM_SLIDE and self.mode=="train" and random.random() < 0.5:
            if (grid_idx[0]!=0 and grid_idx[1]!=0) and (grid_idx[0]!=img.shape[0]//self.img_size[0] and grid_idx[1]!=img.shape[1]//self.img_size[1]):
                rand_pos = [np.random.randint(0, self.img_size[0]//4) - self.img_size[0]//4, np.random.randint(0, self.img_size[1]//4)-self.img_size[1]//4]
            else:
                rand_pos = [0, 0]
        else:
            rand_pos = [0, 0]
        self.rand_pos = rand_pos
        img_grid = img[(grid_idx[0]*self.img_size[0]) + self.slide_pos[0] + rand_pos[0] : ((grid_idx[0]+1)*self.img_size[0]) + self.slide_pos[0] + rand_pos[0],
                        (grid_idx[1]*self.img_size[1]) + self.slide_pos[1] + rand_pos[1] : ((grid_idx[1]+1)*self.img_size[1]) + self.slide_pos[1] + rand_pos[1]]
        mask_grid = mask[(grid_idx[0]*self.img_size[0]) + self.slide_pos[0] + rand_pos[0] : ((grid_idx[0]+1)*self.img_size[0]) + self.slide_pos[0] + rand_pos[0],
                         (grid_idx[1]*self.img_size[1]) + self.slide_pos[1] + rand_pos[1] : ((grid_idx[1]+1)*self.img_size[1]) + self.slide_pos[1] + rand_pos[1]]
        return img_grid/255., mask_grid/255.
    
    def get_masked_img(self, img, mask):
        """ multiply mask to surface_volumes """
        masked_img = None
        for channel in range(img.shape[2]):
            img_channel = img[:,:,channel].reshape(img.shape[0], img.shape[1],1)
            masked = img_channel*mask
            if masked_img is None:
                masked_img = masked.reshape(masked.shape[0], masked.shape[1], 1)
            else:
                masked = masked.reshape(masked.shape[0], masked.shape[1], 1)
                masked_img = np.concatenate([masked_img, masked], axis=2)
        return masked_img
    
    
    def get_all_grid(self):
        """ get all grid indices by img size and grid size
        """
        self.grid_indices = []
        for img in self.imgs:
            self.x_grid_size = (img.shape[0] - self.slide_pos[0]) // self.img_size[0]
            self.y_grid_size = (img.shape[1] - self.slide_pos[1]) // self.img_size[1]
            grid_img = []
            for i in range(self.x_grid_size):
                for j in range(self.y_grid_size):
                    grid_img.append([i, j])
            self.grid_indices.append(grid_img)
        return self.grid_indices
          
    def fileter_grid(self):
        """ get grid indices which mask is not 0 by all grid indices"""
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
        """ get flatten index list by grid indices
            Returns:flatten_grid (list): flatten index list [[img_idx, grid_idx], [img_idx, grid_idx], ...]
        """
        flatten_grid = []
        for img_idx, grid_indices in enumerate(self.grid_indices):
            for grid_idx in grid_indices:
                grid_imgidx_list = [img_idx]
                grid_imgidx_list.extend(grid_idx)
                flatten_grid.append(grid_imgidx_list)
        self.flatten_grid = flatten_grid
        return self.flatten_grid
    
    def channel_shuffle(self, img):
        img = img.transpose(2, 0, 1)
        np.random.shuffle(img)
        return img.transpose(1, 2, 0)

    def __len__(self):
        return len(self.flatten_grid)

    def __getitem__(self, idx):
        # get indices
        img_grid_idx = self.flatten_grid[idx]
        img_idx = img_grid_idx[0]
        grid_idx = img_grid_idx[1:]
        # get img & surface_vol
        mask = self.imgs[img_idx]
        surface_vol = self.surface_vols[img_idx]
        # mask = self.get_grid_img(mask, grid_idx)/255.
        # surface_vol = self.get_grid_img(surface_vol, grid_idx)/255.
        mask, surface_vol = self.get_grid_img_and_mask(mask, surface_vol, grid_idx)
        # multiple small mask 
        assert surface_vol.shape[0]==mask.shape[0] and surface_vol.shape[1]==mask.shape[1] , "surface_vol_list shape is not same as img shape"
        img = surface_vol
        # transform
        if self.mode == "test":
            if self.transform:
                img = self.transform(image=img)["image"]
            else:
                img = img.transpose(2, 0, 1)
                img = torch.tensor(img, dtype=torch.float32)
            return img, grid_idx
        elif self.mode == "train" or self.mode=="valid":
            # get label(segmentation mask)
            label = self.labels[img_idx]
            label = self.get_grid_img(label, grid_idx)
            if self.transform:
                transformed = self.transform(image=img, mask=label)
                img = transformed["image"]
                label = transformed["mask"]
                label = label.permute(2, 0, 1)/255. # (channel, h, w)
            else:
                img = img.transpose(2, 0, 1) # (channel, h, w)
                label = label.transpose(2, 0, 1)/255. # (channel, h, w){}
                img = torch.tensor(img, dtype=torch.float32)
                label = torch.tensor(label, dtype=torch.float32)
            assert img is not None and label is not None, f"img or label is None {img} {label}, {img_idx}, {grid_idx}, {self.rand_pos}"
            return img, label, grid_idx

def train_fn(train_loader, model, criterion, epoch ,optimizer, scheduler, CFG):
    model.train()
    batch_time = AverageMeter()
    losses = AverageMeter()
    start = end = time.time()
    for batch_idx, (images, targets, _) in enumerate(train_loader):
        images = images.to(device, non_blocking = True).float()
        targets = targets.to(device, non_blocking = True).float()     
        preds = model(images)
        preds = TF.resize(img=preds, size=(CFG["input_img_size"][0], CFG["input_img_size"][1]))
        assert preds is not None, f"preds is None, {preds}, {images}, {targets}"
        loss = criterion(preds, targets)
        preds = torch.sigmoid(preds)
        assert loss is not None, f"loss is None, {loss}, {preds}, {targets}"
        losses.update(loss.item(), CFG["batch_size"]) 
        targets = targets.detach().cpu().numpy().ravel().tolist()
        preds = preds.detach().cpu().numpy().ravel().tolist()
        loss.backward() # パラメータの勾配を計算
        optimizer.step() # モデル更新
        optimizer.zero_grad() # 勾配の初期化
                
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % CFG["print_freq"] == 0 or batch_idx == (len(train_loader)-1):
            LOGGER.info('Epoch: [{0}][{1}/{2}] '
                    'Elapsed {remain:s} '
                    'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                    .format(
                        epoch, batch_idx, len(train_loader), batch_time=batch_time, loss=losses,
                        remain=timeSince(start, float(batch_idx+1)/len(train_loader)),
            ))
        del preds, images, targets
    gc.collect()
    torch.cuda.empty_cache()
    return losses.avg

def valid_fn(model, valid_loader, CFG, criterion=None):
    model.eval()# モデルを検証モードに設定
    test_targets = []
    test_preds = []
    test_grid_idx = []
    batch_time = AverageMeter()
    losses = AverageMeter()
    start = end = time.time()
    for batch_idx, (images, targets, grid_idx) in enumerate(valid_loader):
        images = images.to(device, non_blocking = True).float()
        targets = targets.to(device, non_blocking = True).float()
        with torch.no_grad():
            preds = model(images)
            preds = TF.resize(img=preds, size=(CFG["input_img_size"][0], CFG["input_img_size"][1]))
            assert preds is not None, f"preds is None, {preds}, {images}, {targets}"
            if not criterion is None:
                loss = criterion(preds, targets)
                assert loss is not None, f"loss is None, {loss}, {preds}, {targets}"
            preds = torch.sigmoid(preds)
        if not criterion is None:
            losses.update(loss.item(), CFG["batch_size"])
        batch_time.update(time.time() - end)

        targets = targets.detach().cpu().numpy()
        preds = preds.detach().cpu().numpy()
        
        test_preds.extend([preds[idx, :,:,:].transpose(1,2,0) for idx in range(preds.shape[0])])
        test_targets.extend([targets[idx, :,:,:].transpose(1,2,0) for idx in range(targets.shape[0])])
        test_grid_idx.extend([[x_idx, y_idx] for x_idx, y_idx in zip(grid_idx[0].tolist(), grid_idx[1].tolist())])

        if (batch_idx % CFG["print_freq"] == 0 or batch_idx == (len(valid_loader)-1)) and (not criterion is None):
            LOGGER.info('EVAL: [{0}/{1}] '
                'Elapsed {remain:s} '
                'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                .format(
                    batch_idx, len(valid_loader), batch_time=batch_time, loss=losses,
                    remain=timeSince(start, float(batch_idx+1)/len(valid_loader)),
                ))
        del preds, images, targets
        gc.collect()
        torch.cuda.empty_cache()
    if criterion is None:
        return test_targets, test_preds, test_grid_idx
    else:
        return test_targets, test_preds, test_grid_idx, losses.avg

def concat_grid_img(img_list, label_list, grid_idx_list, valid_dir_list, CFG, slide_pos=[0,0], tta="default"):
    # concat pred img and label to original size
    img_path = os.path.join(CFG["TRAIN_DIR"], valid_dir_list[0], "inklabels.png")
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    label_img = img.reshape(img.shape[0], img.shape[1], 1)
    label_img = (label_img > 0).astype(np.float32)
    img = img.reshape(img.shape[0], img.shape[1], 1)
    pred_img = np.zeros_like(img).astype(np.float32)
    for img_idx, grid_idx in enumerate(grid_idx_list):
        img_ = img_list[img_idx]
        img_ = cv2.resize(img_, dsize=(CFG["img_size"][0], CFG["img_size"][1]))
        img_ = img_[:, :, np.newaxis]
        if tta=="default":
            pred_img[grid_idx[0]*CFG["img_size"][0]+slide_pos[0] : (grid_idx[0]+1)*CFG["img_size"][0]+slide_pos[0],
                    grid_idx[1]*CFG["img_size"][1]+slide_pos[1] : (grid_idx[1]+1)*CFG["img_size"][1]+slide_pos[1], :] += img_
        elif tta=="vflip":
            pred_img[grid_idx[0]*CFG["img_size"][0]+slide_pos[0] : (grid_idx[0]+1)*CFG["img_size"][0]+slide_pos[0],
                    grid_idx[1]*CFG["img_size"][1]+slide_pos[1] : (grid_idx[1]+1)*CFG["img_size"][1]+slide_pos[1], :] += np.flipud(img_)
        elif tta=="hflip":
            pred_img[grid_idx[0]*CFG["img_size"][0]+slide_pos[0] : (grid_idx[0]+1)*CFG["img_size"][0]+slide_pos[0],
                    grid_idx[1]*CFG["img_size"][1]+slide_pos[1] : (grid_idx[1]+1)*CFG["img_size"][1]+slide_pos[1], :] += np.fliplr(img_)
        else:
            pred_img[grid_idx[0]*CFG["img_size"][0]+slide_pos[0] : (grid_idx[0]+1)*CFG["img_size"][0]+slide_pos[0],
                    grid_idx[1]*CFG["img_size"][1]+slide_pos[1] : (grid_idx[1]+1)*CFG["img_size"][1]+slide_pos[1], :] += img_
        
    return pred_img, label_img

def save_and_plot_oof(mode, fold, slice_idx, slide_idx, tta, valid_preds_img, valid_targets_img, valid_preds_binary, CFG):
    cv2.imwrite(os.path.join(CFG["OUTPUT_DIR"], "imgs", f"fold{fold}_{mode}_slice{slice_idx}_slide{slide_idx}_{tta}_valid_pred_img.png"), valid_preds_img*255)
    cv2.imwrite(os.path.join(CFG["OUTPUT_DIR"], "imgs", f"fold{fold}_{mode}_slice{slice_idx}_slide{slide_idx}_{tta}_valid_predbin_img.png"), valid_preds_binary*255)
    cv2.imwrite(os.path.join(CFG["OUTPUT_DIR"], "imgs", f"fold{fold}_{mode}_slice{slice_idx}_slide{slide_idx}_{tta}_valid_targets_img.png"), valid_targets_img*255)

def get_tta_aug(aug_type):
    if aug_type=="default":
        return A.Compose([
            A.Resize(CFG["input_img_size"][0], CFG["input_img_size"][1]),
            ToTensorV2(),
            ])
    elif aug_type=="hflip":
        return A.Compose([
            A.Resize(CFG["input_img_size"][0], CFG["input_img_size"][1]),
            A.HorizontalFlip(p=1.0),
            ToTensorV2(),
        ])
    elif aug_type=="vflip":
        return A.Compose([
            A.Resize(CFG["input_img_size"][0], CFG["input_img_size"][1]),
            A.VerticalFlip(p=1.0),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(CFG["input_img_size"][0], CFG["input_img_size"][1]),
            ToTensorV2(),
            ])

def criterion(y_pred, y_true):
    weights = torch.tensor([0.3]).cuda()
    return 0.5*torch.nn.BCEWithLogitsLoss(pos_weight=weights)(y_pred, y_true) + 0.5*smp.losses.TverskyLoss("binary")(y_pred, y_true)

def training_loop(CFG):
    best_score_list = []
    best_threshold_list = []
    best_epoch_list = []
    slice_ave_score_list = []
    slice_ave_auc_list = []
    slice_ave_score_threshold_list = []
    for fold in CFG["folds"]:
        LOGGER.info(f"-- fold{fold} training start --") 
        # set model & learning fn
        model = SegModel(CFG)
        # model = ModifiedPSPNet(CFG)
        # model = smp.PSPNet(encoder_name=CFG["model_name"], 
        #                     encoder_weights="imagenet", 
        #                     classes=CFG["out_channels"], 
        #                     )
        model = model.to(device)
        valid_img_slice = []
        # weights = torch.tensor([0.3]).cuda()
        # criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
        # criterion = torch.nn.BCEWithLogitsLoss()
        # criterion = smp.losses.TverskyLoss("binary")
        optimizer = AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"], amsgrad=False)
        scheduler = CosineAnnealingLR(optimizer, T_max=CFG["T_max"], eta_min=CFG["min_lr"], last_epoch=-1)
        # training
        best_score = -np.inf
        best_auc = -np.inf
        best_valloss = np.inf
        best_auc_valloss = np.inf
        best_threshold = -1
        start_time = time.time()
        best_epoch = -1
        best_auc_epoch = -1
        valid_slice_ave = None       
        for slice_idx, surface_list in enumerate(CFG["SURFACE_LIST"]):
            LOGGER.info(f"surface_list: {surface_list}")
            # separate train/valid data 
            train_dirs = CFG["TRAIN_DIR_LIST"][fold]
            valid_dirs = CFG["VALID_DIR_LIST"][fold]
            train_dataset = VCID_Dataset(CFG, train_dirs, surface_list, mode="train", transform=train_transforms)
            valid_dataset = VCID_Dataset(CFG, valid_dirs, surface_list, mode="valid", transform=valid_transforms)
            train_loader = DataLoader(train_dataset, batch_size=CFG["batch_size"], shuffle = True,
                                        num_workers = CFG["num_workers"], pin_memory = True)
            valid_loader = DataLoader(valid_dataset, batch_size=CFG["batch_size"], shuffle = False,
                                        num_workers = CFG["num_workers"], pin_memory = True)
            for epoch in range(1, CFG["n_epoch"] + 1):
                epochs_ = epoch + CFG["n_epoch"] * slice_idx
                LOGGER.info(f'- epoch:{epochs_} -')
                train_loss_avg = train_fn(train_loader, model, criterion, epochs_ ,optimizer, scheduler, CFG)
                valid_targets, valid_preds, valid_grid_idx, valid_loss_avg = valid_fn(model, valid_loader, CFG, criterion)
                
                # target, predをconcatして元のサイズに戻す
                valid_preds_img, valid_targets_img  = concat_grid_img(valid_preds, valid_targets, valid_grid_idx, valid_dirs, CFG)
                valid_score, valid_threshold, auc, dice_list = calc_cv(valid_targets_img, valid_preds_img)
                valid_preds_binary = (valid_preds_img > valid_threshold).astype(np.uint8)
                
                elapsed = time.time() - start_time
                LOGGER.info(f"\t epoch:{epochs_}, avg train loss:{train_loss_avg:.4f}, avg valid loss:{valid_loss_avg:.4f}")
                LOGGER.info(f"\t score:{valid_score:.4f}(th={valid_threshold:3f}), auc={auc:4f}::: time:{elapsed:.2f}s")
                if not CFG["DEBUG"]:
                    logging_metrics_epoch(CFG, fold, epoch, slice_idx, train_loss_avg, valid_loss_avg, valid_score, valid_threshold, auc)
                scheduler.step()
                # validationスコアがbestを更新したらモデルを保存する
                if valid_score > best_score:
                    best_epoch = epochs_
                    best_valloss = valid_loss_avg
                    best_score = valid_score
                    best_threshold = valid_threshold
                    model_name = CFG["model_name"]
                    model_path = os.path.join(CFG["OUTPUT_DIR"], f'{model_name}_fold{fold}.pth')
                    torch.save(model.state_dict(), model_path) 
                    LOGGER.info(f'Epoch {epochs_} - Save Best Score: {best_score:.4f}.')
                    LOGGER.info(f"dice_list: {dice_list}")
                
                if auc > best_auc:
                    best_auc = auc
                    best_auc_epoch = epochs_
                    best_auc_valloss = valid_loss_avg
                    model_name = CFG["model_name"]
                    model_path = os.path.join(CFG["OUTPUT_DIR"], f'{model_name}_auc_fold{fold}.pth')
                    torch.save(model.state_dict(), model_path) 
                    LOGGER.info(f'Epoch {epochs_} - Save Best AUC: {best_auc:.4f}. (score={valid_score:4f}, thr={valid_threshold:.3f}).Model is saved.')
                    # save oof
            # valid_img_slice.append(valid_preds_img)
            if valid_slice_ave is None:
                valid_slice_ave = valid_preds_img
            else:
                valid_slice_ave += valid_preds_img
        valid_slice_ave /= len(CFG["SURFACE_LIST"])
        valid_sliceave_score, valid_sliceave_threshold, ave_auc, dice_list = calc_cv(valid_targets_img, valid_slice_ave)
        
        slice_ave_score_list.append(valid_sliceave_score)
        slice_ave_auc_list.append(ave_auc)
        slice_ave_score_threshold_list.append(valid_sliceave_threshold)
 
        valid_slice_binary = (valid_slice_ave > valid_sliceave_threshold).astype(np.uint8)
        save_and_plot_oof("average", fold, 999, 999, "train",valid_slice_ave, valid_targets_img, valid_slice_binary, CFG)
        LOGGER.info(f'[fold{fold}] slice ave score:{valid_sliceave_score:.4f}(th={valid_sliceave_threshold:3f}), auc={ave_auc:4f}')
        LOGGER.info(f'[fold{fold}] BEST Epoch {best_epoch} - Save Best Score:{best_score:.4f}. Best loss:{best_valloss:.4f}')
        LOGGER.info(f'[fold{fold}] BEST AUC Epoch {best_auc_epoch} - Save Best Score:{best_auc:.4f}. Best loss:{best_auc_valloss:.4f}')
        
        best_score_list.append(best_score)
        best_threshold_list.append(best_threshold)
        best_epoch_list.append(best_epoch)
        del model, train_loader, train_dataset, valid_loader, valid_dataset, valid_preds_img, valid_targets_img, valid_preds_binary
        gc.collect()
        torch.cuda.empty_cache()
        
    for fold, (best_score, best_threshold, best_epoch) in enumerate(zip(best_score_list, best_threshold_list, best_epoch_list)):
        LOGGER.info(f"fold[{fold}] BEST SCORE = {best_score:.4f} thr={best_threshold} (epoch={best_epoch})")
        LOGGER.info(f"fold[{fold}] slice ave score:{slice_ave_score_list[fold]:.4f}(th={slice_ave_score_threshold_list[fold]:3f}), auc={slice_ave_auc_list[fold]:4f}")
    return best_score_list, best_threshold_list, best_epoch_list


def slide_inference_tta(CFG, tta_list):
    start_time = time.time()
    slice_ave_score_list, slice_ave_auc_list, slice_ave_score_threshold_list = [], [], []
    for fold in CFG["folds"]:
        print(f"-- fold{fold} slide inference start --")
 
        # set model & learning fn
        model = SegModel(CFG)
        # model = ModifiedPSPNet(CFG)
        # model = smp.PSPNet(encoder_name=CFG["model_name"], 
        #                     encoder_weights="imagenet", 
        #                     classes=CFG["out_channels"], 
        #                     )
        model_path = os.path.join(CFG["OUTPUT_DIR"], f'{CFG["model_name"]}_auc_fold{fold}.pth')
        # model_path = os.path.join(CFG["OUTPUT_DIR"], f'{CFG["model_name"]}_fold{fold}.pth')
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        valid_img_slice = None
        for slice_idx, surface_list in enumerate(CFG["SURFACE_LIST"]):
            print("surface_list: ", surface_list)
            surface_volumes = None
            for slide_pos in CFG["slide_pos_list"]:
                print("slide pos:", slide_pos)
                valid_dirs = CFG["VALID_DIR_LIST"][fold]
                for tta in tta_list:
                    print(f"tta:{tta}")
                    valid_transforms = get_tta_aug(tta)
                    valid_dataset = VCID_Dataset(CFG, valid_dirs, surface_list, surface_volumes, slide_pos,
                                                 mode="valid", transform=valid_transforms)
                    surface_volumes = valid_dataset.get_surface_volumes()
                    valid_loader = DataLoader(valid_dataset, batch_size=CFG["batch_size"], shuffle = False,
                                                num_workers = CFG["num_workers"], pin_memory = True)

                    valid_targets, valid_preds, valid_grid_idx = valid_fn(model, valid_loader, CFG)

                    # target, predをconcatして元のサイズに戻す
                    valid_preds_img, valid_targets_img  = concat_grid_img(valid_preds, valid_targets, valid_grid_idx, valid_dirs, CFG, slide_pos, tta)
                    valid_score, valid_threshold, auc, dice_list = calc_cv(valid_targets_img, valid_preds_img)
                    valid_preds_binary = (valid_preds_img > valid_threshold).astype(np.uint8)
                    # save_and_plot_oof("slide_tta", fold, slice_idx, valid_preds_img, valid_targets_img, valid_preds_binary) 

                    elapsed = time.time() - start_time
                    print(f"\t score:{valid_score:.4f}(th={valid_threshold:3f}), auc={auc:4f}::: time:{elapsed:.2f}s")
                    if valid_img_slice is None:
                        valid_img_slice = valid_preds_img
                    else:
                        valid_img_slice += valid_preds_img

        valid_img_slice /= (len(CFG["SURFACE_LIST"])*len(CFG["slide_pos_list"])*len(tta_list))
        valid_sliceave_score, valid_sliceave_threshold, ave_auc, dice_list = calc_cv(valid_targets_img, valid_img_slice)
        
        slice_ave_score_list.append(valid_sliceave_score)
        slice_ave_auc_list.append(ave_auc)
        slice_ave_score_threshold_list.append(valid_sliceave_threshold)

        valid_slice_binary = (valid_img_slice > valid_sliceave_threshold).astype(np.uint8)
        # save_and_plot_oof("average_tta", fold, 555, valid_img_slice, valid_targets_img, valid_slice_binary)
        cv2.imwrite(os.path.join(CFG["OUTPUT_DIR"], "imgs", f"fold{fold}_oofpred.png"), valid_img_slice*255)
        print(f'[fold{fold}] slice ave score:{valid_sliceave_score:.4f}(th={valid_sliceave_threshold:3f}), auc={ave_auc:4f}')
         
        del model, valid_loader, valid_dataset, valid_preds_img, valid_targets_img, valid_preds_binary
        gc.collect()
        torch.cuda.empty_cache()
    return slice_ave_score_list, slice_ave_auc_list, slice_ave_score_threshold_list


def oof_score_check(CFG):
    pred_flatten_list = []
    mask_flatten_list = []
    for fold in CFG["folds"]:
        # pred_path = os.path.join(CFG["OUTPUT_DIR"], "imgs", f"fold{fold}_average_slice555_slide555_valid_pred_img.png")
        # mask_path = os.path.join(CFG["OUTPUT_DIR"], "imgs", f"fold{fold}_average_slice555_slide555_valid_targets_img.png")
        pred_path = os.path.join(CFG["OUTPUT_DIR"], "imgs", f"fold{fold}_oofpred.png")
        mask_path = os.path.join(CFG["TRAIN_DIR"], str(CFG["VALID_DIR_LIST"][fold][0]), "inklabels.png")
        LOGGER.info(pred_path)
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred_flatten_list.extend(pred_img.flatten())
        mask_flatten_list.extend(mask_img.flatten())

    mask_flatten_list = np.array(mask_flatten_list)/255.
    mask = np.array(mask_flatten_list).astype(int)
    pred = np.array(pred_flatten_list)/255.
    for th in np.array(range(10, 100+1, 5)) / 100:
        dice = fbeta_numpy(mask, (pred >= th).astype(int), beta=0.5)
        LOGGER.info(f"th={th:.2f}, dice={dice:.4f}")
        wandb.log({"ALL FOLD OOF SCORE" : {"threshold":th, 
                                           "dice":dice,
                                           }
                   })

def oofscore_log(CFG):
    for fold in CFG["folds"]:
        LOGGER.info(f"fold[{fold}] slice ave score:{slice_ave_score_list[fold]:.4f}(th={slice_ave_score_threshold_list[fold]:3f}), auc={slice_ave_auc_list[fold]:4f}")
        wandb.log({"OOF SCORE" : {f"slice average score":slice_ave_score_list[fold],
                                f"slice average threshold":slice_ave_score_threshold_list[fold],
                                f"slice_average auc":slice_ave_auc_list[fold],
                                "fold":fold,
                                }})

if __name__=="__main__":
    LOGGER.info(CFG)
    if not CFG["DEBUG"]:
        WANDB_CONFIG = {'competition': 'vcid', '_wandb_kernel': 'taro'}
        os.environ["WANDB_SILENT"] = "true"
        wandb.init(project=WANDB_CONFIG["competition"], config=CFG, group=CFG["EXP_CATEGORY"], name=CFG["EXP_NAME"], reinit=True)

    best_score_list, best_threshold_list, best_epoch_list = training_loop(CFG)
    if not CFG["DEBUG"]:
        tta_list = ["default", "hflip", "vflip"]
    else:
        tta_list = ["default"]
    slice_ave_score_list, slice_ave_auc_list, slice_ave_score_threshold_list = slide_inference_tta(CFG, tta_list) 
    LOGGER.info(f"scores mean:{np.mean(slice_ave_score_list):.4f}(th={np.mean(slice_ave_score_threshold_list):3f}), auc={np.mean(slice_ave_auc_list):4f}") 
    if not CFG["DEBUG"]:
        oofscore_log(CFG)
        oof_score_check(CFG)
        wandb.finish()
        

    




