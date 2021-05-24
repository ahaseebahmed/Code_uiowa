
import miscTorch as sf
import torch
import torch.nn as nn
import torch.nn.functional as F


class layer (nn.Module):
    def __init__(self,inpch,outch):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=inpch,out_channels=outch,kernel_size=3,
                            padding=1,bias=False,padding_mode='replicate')
        self.bn=nn.BatchNorm2d(outch)
    
    def forward(self,x):
        x=self.conv1(x)
        x=self.bn(x)
        x=F.relu(x)
        return x


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lay1=layer(2,32)
        self.lay2=layer(32,64)
        self.lay3=layer(64,64)
        self.lay4=layer(64,64)
        self.lay5=layer(64,32)
        self.lay6=nn.Conv2d(32,2,3,1,padding=1,bias=False)


    def forward(self,inp):
        inp,mn,st=sf.pt_normalizeMeanStd(inp)
        x=self.lay1(inp)
        x=self.lay2(x)
        x=self.lay3(x)
        x=self.lay4(x)
        x=self.lay5(x)
        x=self.lay6(x)
        x1=x+inp
        x1=x1*st+mn
        
        return x1

