from torch import nn
import torch.nn.functional as F

def ConvLReLU(in_ch, out_ch, kernel_size, dilation):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=dilation, dilation=dilation),
        nn.LeakyReLU(inplace=True)
    )

class FewShotCNN(nn.Module):
    def __init__(self, in_ch, n_class, size='S'):
        super().__init__()
        
        assert size in ['S', 'M', 'L']
        
        dilations = {
            'S': [1, 2, 1, 2, 1],
            'M': [1, 2, 4, 1, 2, 4, 1],
            'L': [1, 2, 4, 8, 1, 2, 4, 8, 1],            
        }[size]
        
        channels = {
            'S': [128, 64, 64, 32],
            'M': [128, 64, 64, 64, 64, 32],
            'L': [128, 64, 64, 64, 64, 64, 64, 32],         
        }[size]
        channels = [in_ch] + channels + [n_class]
        
        layers = []
        for d, c_in, c_out in zip(dilations, channels[:-1], channels[1:]):
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=d, dilation=d))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers[:-1])
        
    def forward(self, x):
        return self.layers(x)
    
    
# U-Net code modified from https://amaarora.github.io/2020/09/13/unet.html
class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3)
        self.relu  = nn.ReLU()
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3)
    
    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class Encoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.enc_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)])
        self.pool       = nn.MaxPool2d(2)
    
    def forward(self, x):
        ftrs = []
        for block in self.enc_blocks:
            x = block(x)
            ftrs.append(x)
            x = self.pool(x)
        return ftrs


class Decoder(nn.Module):
    def __init__(self, chs):
        super().__init__()
        self.chs         = chs
        self.upconvs    = nn.ModuleList([nn.ConvTranspose2d(chs[i], chs[i+1], 2, 2) for i in range(len(chs)-1)])
        self.dec_blocks = nn.ModuleList([Block(chs[i], chs[i+1]) for i in range(len(chs)-1)]) 
        
    def forward(self, x, encoder_features):
        for i in range(len(self.chs)-1):
            x        = self.upconvs[i](x)
            enc_ftrs = self.crop(encoder_features[i], x)
            x        = torch.cat([x, enc_ftrs], dim=1)
            x        = self.dec_blocks[i](x)
        return x
    
    def crop(self, enc_ftrs, x):
        _, _, H, W = x.shape
        enc_ftrs   = torchvision.transforms.CenterCrop([H, W])(enc_ftrs)
        return enc_ftrs


class UNet(nn.Module):
    def __init__(
        self, 
        enc_chs=(3,64,128,256,512,1024), 
        dec_chs=(1024, 512, 256, 128, 64), 
        n_class=1, 
        retain_dim=False, 
    ):   
        super().__init__()
        self.encoder     = Encoder(enc_chs)
        self.decoder     = Decoder(dec_chs)
        self.head        = nn.Conv2d(dec_chs[-1], n_class, 1)
        self.retain_dim  = retain_dim

    def forward(self, x):
        enc_ftrs = self.encoder(x)
        out      = self.decoder(enc_ftrs[::-1][0], enc_ftrs[::-1][1:])
        out      = self.head(out)
        return out