import torch
import torch.nn as nn
import torch.nn.functional as F

#Padding convolutions since I want HxW Input = HxW Output

#Batchnorm is not part of the original paper but improves the result

def double_conv(in_channels, out_channels):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return conv

    
class CamvidUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CamvidUNet, self).__init__()
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.down1 = double_conv(3, 64)
        self.down2 = double_conv(64, 128)
        self.down3 = double_conv(128, 256)
        self.down4 = double_conv(256, 512)
        self.down5 = double_conv(512, 1024)
        
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.upconv1 = double_conv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.upconv2 = double_conv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv3 = double_conv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv4 = double_conv(128, 64)
        
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
        
    
    def forward(self, x):
        #Encoder Part
        x1 = self.down1(x) #skip
        x2 = self.pool(x1)
        x3 = self.down2(x2) #skip
        x4 = self.pool(x3)
        x5 = self.down3(x4) #skip
        x6 = self.pool(x5)
        x7 = self.down4(x6) #skip
        x8 = self.pool(x7)
        x9 = self.down5(x8) 
        
        #Decoder part
        x = self.up1(x9)
        x = self.upconv1(torch.cat([x7, x], dim=1))
        x = self.up2(x)
        x = self.upconv2(torch.cat([x5, x], dim=1))
        x = self.up3(x)
        x = self.upconv3(torch.cat([x3, x], dim=1))
        x = self.up4(x)
        x = self.upconv4(torch.cat([x1, x], dim=1))
        output = self.out(x)
        
        return(output)