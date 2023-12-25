import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
# input spatial size of W, stride of S, padding of P and filter (kernel) size of F 
# we can calculate the output shape as floor((W - F + 2P)/S + 1)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), #Same size conv --> floor((W - 3 + 2)/1 +1) == floor(W-1+1) == w
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, features=[64, 128, 256, 512]):
        super().__init__()
        self.uplis = nn.ModuleList()
        self.downlis = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)           #Downsamples the HxW
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.segment = nn.Conv2d(features[0], out_channels, 1)      #1x1 conv for final segmentation layer

        #Encoder part of the UNET
        for c in features:
            self.downlis.append(DoubleConv(in_channels, c))
            in_channels= c

        #Decoder part of the UNET
        for c in reversed(features):
            self.uplis.append(nn.ConvTranspose2d(c*2, c, kernel_size=2, stride=2))
            self.uplis.append(DoubleConv(c*2, c))

    def forward(self, x):
        skip_lis = []

        for down in self.downlis:
            x = down(x)
            skip_lis.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_lis = skip_lis[::-1]

        for i in range(0, len(self.uplis), 2):
            x = self.uplis[i](x)
            skip_to_cat = skip_lis[i//2]

            if x.shape != skip_to_cat.shape:                    #Resizing if required before concatenating
                x = TF.resize(x, size= skip_to_cat.shape[2:])

            concatenated = torch.cat((skip_to_cat, x), dim=1)   #concatenates along channel dim
            x = self.uplis[i+1](concatenated)
        
        return self.segment(x)


# Testing for shapes
if __name__ == "__main__":
    x = torch.randn((3, 1, 161, 161))
    model = UNET(in_channels=1, out_channels=2)
    pred = model(x)
    print (pred.shape[2:] == x.shape[2:])