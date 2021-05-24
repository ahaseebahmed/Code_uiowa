
import supportingFun as sf
import torch
import torch.nn as nn
import torch.nn.functional as F


class layer (nn.Module):
    def __init__(self,inpch,outch):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=inpch,out_channels=outch,kernel_size=3,
                            padding=1,bias=False)
        self.bn=nn.BatchNorm2d(outch)
    
    def forward(self,x):
        x=self.conv1(x)
        #x=self.bn(x)
        x=F.relu(x)
        return x


class SmallModel1(nn.Module):
    def __init__(self):
        super().__init__()
        self.lay1=layer(60,256)
        #self.lay2=layer(256,256)
        self.lay3=layer(256,512)
        self.lay4=layer(512,1024)
        self.lay5=layer(1024,1024)
        self.lay7=layer(1024,512)
        #self.lay7=layer(512,512)
        self.lay8=layer(512,256)
        #self.lay9=layer(256,256)
        self.lay10=nn.Conv2d(256,60,3,1,padding=1,bias=False)#layer(128,60)


    def forward(self,inp):
        #inp,mn,st=sf.pt_normalizeMeanStd(inp)
        x=self.lay1(inp)
        #x=self.lay2(x)
        x=self.lay3(x)
        x=self.lay4(x)
        x=self.lay5(x)
        x=self.lay7(x)
        #x=self.lay7(x)
        x=self.lay8(x)
        #x=self.lay9(x)
        x=self.lay10(x)
        x1=x+inp
        #x1=x1*st+mn
        
        return x1
