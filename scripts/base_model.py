
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
        self.encoder = timm.create_model(CFG["model_name"], in_chans=CFG["inp_channels"], features_only=True, out_indices=[0,1,2,3], pretrained=CFG["pretrained"])
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
    def __init__(self, in_channels, out_channels):
        super().__init__():
        """
        encoderから320channelsで来たら、upconvしたものと合わせると、
        倍になるのでそれをchannelの入力にする？？
        """
        self.UpConv_1 = UpConv(320, 160)
        self.UpConv_2 = UpConv(160, 80)
        self.UpConv_3 = UpConv(80, 20)
        self.UpConv_5 = UpConv(24, 16)
    
    def forward(self, skip_connection_list):
        emb = self.UpConv_1(skip_connection_list[3]) # emb.shape = (None, 160, 14, 14)
        print("1st emb", emb.shape)
        emb_cat = torch.cat([skip_connection_list[2], emb], dim = 1)
        emb = self.UpConv_2(emb_cat)
        emb_cat = torch.cat([])



if __name__=="__main__":
    CFG = {
        "model_name" : "tf_efficientnet_b0",
        "inp_channels" : 3,
        "num_output" : 1,
        "pretrained" : True
    }
    model = NFLNet(CFG)
    # summary(
    #     model,
    #     input_size=(batch_size, 3, 224, 224),
    #     col_names=["output_size", "num_params"],
    #     depth=5,
    # )
    # feat_ext = create_feature_extractor(model, {"Conv2d":"skip_connection_1"})
    x = torch.rand((1, 3, 224, 224)).float()
    out_feat = model(x)
    [print(o.shape) for o in out_feat]