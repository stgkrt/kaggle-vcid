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
DEBUG = False 
EXP_NAME = "exp_stack_009"
EXP_YAML_PAHT = os.path.join("/working", "output", EXP_NAME, "Config.yaml")
# read yaml file to CFG
with open(EXP_YAML_PAHT) as yaml_file:
    CFG = yaml.load(yaml_file, Loader=yaml.FullLoader)
os.makedirs(os.path.join("/working", "output", EXP_NAME, "imgs"), exist_ok=True)

CFG["EXP_NAME"] = EXP_NAME
CFG["DEBUG"] = DEBUG
CFG["OUTPUT_DIR"] = os.path.join("/working", "output", EXP_NAME)
CFG["SUMMARY"] = f"{EXP_NAME}: model:efficientnetb0, stacking"

if DEBUG:
    CFG["n_epoch"] = 10
    CFG["folds"] = [0]
    CFG["slide_pos_list"] = [[0,0], [0, 128], [128, 0], [128, 128]]
print(CFG["pred_names"])

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
class Encoder(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.encoder = timm.create_model(CFG["model_name"], in_chans=CFG["inp_channels"], 
                                         features_only=True, out_indices=CFG["out_indices"], pretrained=CFG["pretrained"])
    def forward(self, img):
        skip_connection_list = self.encoder(img)
        return skip_connection_list

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 2, padding="same")
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.up(x)
        x = self.bn1(x)
        x = self.conv(x)
        x = self.bn2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.UpConv_0 = UpConv(CFG["channel_nums"][0], CFG["channel_nums"][1])
        self.UpConv_1 = UpConv(CFG["channel_nums"][1]*2, CFG["channel_nums"][2])
        self.UpConv_2 = UpConv(CFG["channel_nums"][2]*2, CFG["channel_nums"][3])
        self.UpConv_3 = UpConv(CFG["channel_nums"][3]*2, CFG["channel_nums"][4])
        self.UpConv_4 = UpConv(CFG["channel_nums"][4]*2, CFG["channel_nums"][4])
    
    def forward(self, skip_connection_list):
        emb = self.UpConv_0(skip_connection_list[-1])
        emb_cat = torch.cat([skip_connection_list[-2], emb], dim = 1)
        
        emb = self.UpConv_1(emb_cat)
        emb_cat = torch.cat([skip_connection_list[-3], emb], dim = 1)
        
        emb = self.UpConv_2(emb_cat)
        emb_cat = torch.cat([skip_connection_list[-4], emb], dim = 1)
        
        emb = self.UpConv_3(emb_cat)
        emb_cat = torch.cat([skip_connection_list[-5], emb], dim = 1)

        emb = self.UpConv_4(emb_cat)
        return emb

class SegModel(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.encoder = Encoder(CFG)
        self.decoder = Decoder(CFG)
        self.head = nn.Sequential(
            nn.Conv2d(CFG["channel_nums"][-1], CFG["out_channels"], kernel_size=1, stride=1, padding=0),
        )
    def forward(self, img):
        skip_connection_list = self.encoder(img)
        emb = self.decoder(skip_connection_list)
        output = self.head(emb)
        return output

# model = SegModel(CFG)
# x = torch.randn(1, CFG["inp_channels"], CFG["input_img_size"][0], CFG["input_img_size"][1])
# print(x.shape)
# out = model(x)
# print(out.shape)

# raise Exception("stop")

""" 
transfomrs
"""
train_transforms = A.Compose([
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
    A.RandomCrop(int(CFG["img_size"][0]*0.8), int(CFG["img_size"][1]*0.8), p=0.3),
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
    def __init__(self, CFG, data_dir_list, pred_names, slide_pos=[0,0], mode="train", transform=None):
        # get config
        self.mode = mode
        self.RANDOM_SLIDE = CFG["RANDOM_SLIDE"]
        self.img_size = CFG["img_size"]
        if self.mode=="train":  self.DATADIR = CFG["TRAIN_DIR"]
        elif self.mode=="valid":    self.DATADIR = CFG["TRAIN_DIR"]
        elif self.mode == "test":   self.DATADIR = CFG["TEST_DIR"]
        self.data_dir_list = data_dir_list
        self.pred_names = pred_names
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
        
        self.oof_preds = self.read_oofpreds()
       
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

    def read_oofpreds(self):
        """ read oof predictions by pred names 
            Returns:oof pred img (concat channel axis) : array(h,w,channel=pred num)
        """
        
        oof_preds_list = []
        for img_idx in self.data_dir_list:
            oof_preds = None
            for pred_name in self.pred_names:
                pred_path = os.path.join(CFG["OOF_DIR"], pred_name + '_' + img_idx + '.npy')
                pred = np.load(pred_path)
                pred = pred.reshape(pred.shape[0], pred.shape[1], 1)
                if oof_preds is None:
                    oof_preds = pred
                else:
                    oof_preds = np.concatenate([oof_preds, pred], axis=2)
            oof_preds_list.append(oof_preds)
        # print("reading[dd] surface volume...")
        return oof_preds_list


    def get_grid_img(self, img, grid_idx):
        """ crop grid img from original img"""
        img_grid = img[(grid_idx[0]*self.img_size[0]) + self.slide_pos[0] : ((grid_idx[0]+1)*self.img_size[0]) + self.slide_pos[0],
                        (grid_idx[1]*self.img_size[1]) + self.slide_pos[1] : ((grid_idx[1]+1)*self.img_size[1]) + self.slide_pos[1]]
        return img_grid
    
    def get_grid_img_and_mask(self, img, mask, grid_idx):
        """ crop grid img from original img"""
        if self.RANDOM_SLIDE and self.mode=="train" and random.random() < 0.8:
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
        return img_grid/255., mask_grid
    
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
    
    def __len__(self):
        return len(self.flatten_grid)

    def __getitem__(self, idx):
        # get indices
        img_grid_idx = self.flatten_grid[idx]
        img_idx = img_grid_idx[0]
        grid_idx = img_grid_idx[1:]
        # get img & surface_vol
        mask = self.imgs[img_idx]
        img = self.oof_preds[img_idx]
        mask, img = self.get_grid_img_and_mask(mask, img, grid_idx)
        # multiple small mask 
        assert img.shape[0]==mask.shape[0] and img.shape[1]==mask.shape[1] , "oofpred shape is not same as img shape"
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
            # if self.mode == "train":
            #     img = self.channel_shuffle(img)
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



# valid_dirs = CFG["VALID_DIR_LIST"][0]
# print("dataset")
# dataset = VCID_Dataset(CFG, valid_dirs, CFG["pred_names"], mode="train", transform=train_transforms)
# print("dataloader")
# dataloader_notrans = DataLoader(dataset, 4, shuffle=False, num_workers=0)

# imshow_batch = 10
# for batch_idx, (imgs, labels, grid_idx) in enumerate(dataloader_notrans):
#     print(imgs.shape)
#     print(labels.shape)
#     if batch_idx == imshow_batch:
#         break

# raise Exception("stop")    


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
    label_path = os.path.join(CFG["TRAIN_DIR"], valid_dir_list[0], "inklabels.png")
    label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    label_img = label_img.reshape(label_img.shape[0], label_img.shape[1], 1)
    label_img = (label_img > 0).astype(np.float32)
    pred_img = np.zeros_like(label_img).astype(np.float32)
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


def save_and_plot_oof(mode, fold, slice_idx, valid_preds_img, valid_targets_img, valid_preds_binary, CFG):
    cv2.imwrite(os.path.join(CFG["OUTPUT_DIR"], "imgs", f"fold{fold}_{mode}_slice{slice_idx}_valid_pred_img.png"), valid_preds_img*255)
    cv2.imwrite(os.path.join(CFG["OUTPUT_DIR"], "imgs", f"fold{fold}_oofpred.png"), valid_preds_img*255)

def get_tta_aug(aug_type):
    if aug_type=="default":
        return A.Compose([ToTensorV2(),])
    elif aug_type=="hflip":
        return A.Compose([
            A.HorizontalFlip(p=1.0),
            ToTensorV2(),
        ])
    elif aug_type=="vflip":
        return A.Compose([
            A.VerticalFlip(p=1.0),
            ToTensorV2(),
        ])
    else:
        return A.Compose([ToTensorV2(),])
      
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
        model = model.to(device)
        valid_img_slice = []
        weights = torch.tensor([0.3]).cuda()
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=weights)
        # criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"], amsgrad=False)
        # scheduler = CosineAnnealingLR(optimizer, T_max=CFG["T_max"], eta_min=CFG["min_lr"], last_epoch=-1)
        scheduler = ExponentialLR(optimizer, gamma=CFG["gamma"], last_epoch=-1)
        
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
        # separate train/valid data 
        train_dirs = CFG["TRAIN_DIR_LIST"][fold]
        valid_dirs = CFG["VALID_DIR_LIST"][fold]
        train_dataset = VCID_Dataset(CFG, train_dirs, CFG["pred_names"], mode="train", transform=train_transforms)
        valid_dataset = VCID_Dataset(CFG, valid_dirs, CFG["pred_names"], mode="valid", transform=valid_transforms)
        train_loader = DataLoader(train_dataset, batch_size=CFG["batch_size"], shuffle = True,
                                    num_workers = CFG["num_workers"], pin_memory = True)
        valid_loader = DataLoader(valid_dataset, batch_size=CFG["batch_size"], shuffle = False,
                                    num_workers = CFG["num_workers"], pin_memory = True)
        for epoch in range(1, CFG["n_epoch"] + 1):
            epochs_ = epoch
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
                logging_metrics_epoch(CFG, fold, epoch, 0, train_loss_avg, valid_loss_avg, valid_score, valid_threshold, auc)
            scheduler.step()
            # validationスコアがbestを更新したらモデルを保存する
            if valid_score > best_score:
                best_epoch = epochs_
                best_valloss = valid_loss_avg
                best_score = valid_score
                best_threshold = valid_threshold
                LOGGER.info(f'Epoch {epochs_} - Save Best Score: {best_score:.4f}. Model is saved.')
                LOGGER.info(f"dice_list: {dice_list}")
            
            if auc > best_auc:
                best_auc = auc
                best_auc_epoch = epochs_
                best_auc_valloss = valid_loss_avg
                model_name = CFG["model_name"]
                model_path = os.path.join(CFG["OUTPUT_DIR"], f'stacking_fold{fold}.pth')
                torch.save(model.state_dict(), model_path) 
                LOGGER.info(f'Epoch {epochs_} - Save Best AUC: {best_auc:.4f}. Model is saved.')
                valid_auc_img = valid_preds_img.copy()
        valid_sliceave_score, valid_sliceave_threshold, ave_auc, dice_list = calc_cv(valid_targets_img, valid_auc_img)
        
        slice_ave_score_list.append(valid_sliceave_score)
        slice_ave_auc_list.append(ave_auc)
        slice_ave_score_threshold_list.append(valid_sliceave_threshold)
 
        valid_slice_binary = (valid_auc_img > valid_sliceave_threshold).astype(np.uint8)
        save_and_plot_oof("average", fold, 999, valid_auc_img, valid_targets_img, valid_slice_binary, CFG)
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


def slide_inference_tta(CFG):
    # tta_list = ["defalt", "hflip", "vflip"]
    tta_list = ["defalt"]
    start_time = time.time()
    slice_ave_score_list, slice_ave_auc_list, slice_ave_score_threshold_list = [], [], []
    for fold in CFG["folds"]:
        LOGGER.info(f"-- fold{fold} slide inference start --")
 
        # set model & learning fn
        model = SegModel(CFG)
        # model_path = os.path.join(CFG["OUTPUT_DIR"], f'{CFG["model_name"]}_fold{fold}.pth')
        model_path = os.path.join(CFG["OUTPUT_DIR"], f'stacking_fold{fold}.pth')
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)
        valid_img_slice = None
        for slide_pos in CFG["slide_pos_list"]:
            LOGGER.info(f"slide pos: {slide_pos}")
            valid_dirs = CFG["VALID_DIR_LIST"][fold]
            for tta in tta_list:
                valid_transforms = get_tta_aug(tta)
                valid_dataset = VCID_Dataset(CFG, valid_dirs, CFG["pred_names"], slide_pos, mode="valid", transform=valid_transforms)
                valid_loader = DataLoader(valid_dataset, batch_size=CFG["batch_size"], shuffle = False,
                                            num_workers = CFG["num_workers"], pin_memory = True)

                valid_targets, valid_preds, valid_grid_idx = valid_fn(model, valid_loader, CFG)
                
                # target, predをconcatして元のサイズに戻す
                valid_preds_img, valid_targets_img  = concat_grid_img(valid_preds, valid_targets, valid_grid_idx, valid_dirs, CFG, slide_pos, tta)
                valid_score, valid_threshold, auc, dice_list = calc_cv(valid_targets_img, valid_preds_img)
                valid_preds_binary = (valid_preds_img > valid_threshold).astype(np.uint8)
                
                elapsed = time.time() - start_time
                LOGGER.info(f"\t score:{valid_score:.4f}(th={valid_threshold:3f}), auc={auc:4f}::: time:{elapsed:.2f}s")
                # valid_img_slice.append(valid_preds_img)
                if valid_img_slice is None:
                    valid_img_slice = valid_preds_img
                else:
                    valid_img_slice += valid_preds_img
        valid_img_slice /= (len(CFG["slide_pos_list"])*len(tta_list))
        valid_sliceave_score, valid_sliceave_threshold, ave_auc, dice_list = calc_cv(valid_targets_img, valid_img_slice)
        LOGGER.info("with gaussian blur score")
        valid_img_blur = cv2.GaussianBlur(valid_img_slice, (65, 65), 32)
        valid_blur_score, valid_slice_blur_threshold, ave_blur_auc, dice_list = calc_cv(valid_targets_img, valid_img_blur)
        
        slice_ave_score_list.append(valid_sliceave_score)
        slice_ave_auc_list.append(ave_auc)
        slice_ave_score_threshold_list.append(valid_sliceave_threshold)

        valid_slice_binary = (valid_img_slice > valid_sliceave_threshold).astype(np.uint8)
        save_and_plot_oof("average", fold, 555, valid_img_slice, valid_targets_img, valid_slice_binary, CFG)
        LOGGER.info(f'[fold{fold}] slice ave score:{valid_sliceave_score:.4f}(th={valid_sliceave_threshold:3f}), auc={ave_auc:4f}') 
        LOGGER.info(f'[fold{fold}] slice ave score(blur):{valid_blur_score:.4f}(th={valid_slice_blur_threshold:3f}), auc={ave_blur_auc:4f}')
        del model, valid_loader, valid_dataset, valid_preds_img, valid_targets_img, valid_preds_binary
        gc.collect()
        torch.cuda.empty_cache()
    return slice_ave_score_list, slice_ave_auc_list, slice_ave_score_threshold_list


def oof_score_check(CFG):
    pred_flatten_list = []
    mask_flatten_list = []
    for fold in CFG["folds"]:
        pred_path = os.path.join(CFG["OUTPUT_DIR"], "imgs", f"fold{fold}_average_slice555_valid_pred_img.png")
        # mask_path = os.path.join(CFG["OUTPUT_DIR"], "imgs", f"fold{fold}_average_slice555_valid_targets_img.png")
        valid_dirs = CFG["VALID_DIR_LIST"][fold]
        mask_path = os.path.join(CFG["TRAIN_DIR"], valid_dirs[0], "inklabels.png")
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
    slice_ave_score_list, slice_ave_auc_list, slice_ave_score_threshold_list = slide_inference_tta(CFG)
    
    
    if not CFG["DEBUG"]:
        oofscore_log(CFG)
        oof_score_check(CFG)
        wandb.finish()
        

    




