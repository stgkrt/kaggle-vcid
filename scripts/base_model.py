
# basic torch and model functions
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision.models.feature_extraction import create_feature_extractor
from torchinfo import summary

# -------------------------------
# Model(timm encoder : https://timm.fast.ai/create_model)
# Efficinet b0 model features
# skip_connection_list[0]# output shape = 1, 16, 112, 112
# skip_connection_list[1]# output shape = 1, 24, 56, 56
# skip_connection_list[2]# output shape = 1, 40, 28, 28
# skip_connection_list[3]# output shape = 1, 320, 7, 7
# -------------------------------
class Encoder(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.encoder = timm.create_model(CFG["model_name"], in_chans=CFG["inp_channels"], features_only=True, out_indices=CFG["out_indices"], pretrained=CFG["pretrained"])
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
    def __init__(self):
        super().__init__()
        """
        encoderから320channelsで来たら、upconvしたものと合わせると、
        倍になるのでそれをchannelの入力にする？？
        """
        channel_nums = [320, 112, 40, 24, 16]
        self.UpConv_0 = UpConv(channel_nums[0], channel_nums[1])
        self.UpConv_1 = UpConv(channel_nums[1]*2, channel_nums[2])
        self.UpConv_2 = UpConv(channel_nums[2]*2, channel_nums[3])
        self.UpConv_3 = UpConv(channel_nums[3]*2, channel_nums[4])
    
    def forward(self, skip_connection_list):
        [print("skip connetction shape", skc.shape) for skc in skip_connection_list]
        emb = self.UpConv_0(skip_connection_list[4]) # emb.shape = (None, 160, 14, 14)
        emb_cat = torch.cat([skip_connection_list[3], emb], dim = 1)
        print("0 emb", emb.shape)
        print("0 emb cat", emb_cat.shape)
        
        emb = self.UpConv_1(emb_cat)
        emb_cat = torch.cat([skip_connection_list[2], emb], dim = 1)
        print("1 emb", emb.shape)
        print("1 emb cat", emb_cat.shape)
        
        emb = self.UpConv_2(emb_cat)
        emb_cat = torch.cat([skip_connection_list[1], emb], dim = 1)
        print("2 emb", emb.shape)
        print("2 emb cat", emb_cat.shape)
        
        emb = self.UpConv_3(emb_cat)
        emb_cat = torch.cat([skip_connection_list[0], emb], dim = 1)
        print("3 emb", emb.shape)
        print("3 emb cat", emb_cat.shape)
        
        return emb_cat

class SegModel(nn.Module):
    def __init__(self, CFG):
        super().__init__()
        self.encoder = Encoder(CFG)
        self.decoder = Decoder()
        self.head = nn.Sequential(
            nn.Conv2d(32, CFG["out_channels"], kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )
    def forward(self, img):
        skip_connection_list = self.encoder(img)
        emb = self.decoder(skip_connection_list)
        output = self.head(emb)
        return output

if __name__=="__main__":
    CFG = {
        "model_name" : "tf_efficientnet_b0",
        "inp_channels" : 3,
        "out_channels" : 1,
        "pretrained" : true,
        "out_indices" : [0,1,2,3,4],
        "batch_size" : 1,
    }

    model = SegModel(CFG)
    x = torch.rand((1, 3, 224, 224)).float()
    out_feat = model(x)
    print(out_feat.shape)
    summary(
        model,
        input_size=(CFG["batch_size"], 3, 224, 224),
        col_names=["output_size", "num_params"],
        depth=5,
    )