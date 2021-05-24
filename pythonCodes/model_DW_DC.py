
import miscTorch as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset



class D_W (nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(),
            nn.ReLU())
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(),
            nn.ReLU())
    
    def forward(self,inp):
        inp,mn,st=sf.pt_normalizeMeanStd(inp)
        x=self.layer1(inp)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.layer4(x)
        x1=x+inp
        x1=x1*st+mn
        return x1


class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.dw=D_W()

    def forward(self,inp):
        x,mn,st=sf.pt_normalizeMeanStd(inp)
        for i in range(3):
            x=atb+lam*net1(x)
            x=sf.D_C(x)
            
    def net1(self,x):        
        y=[]
        x=sf.bas2img(x,D)
        trnDs=TensorDataset(x)
        ldr=DataLoader(trnDs,batch_size=1,shuffle=True,num_workers=8,pin_memory=True)
        for i, (data) in enumerate(ldr):
            data=data.to(gpu=torch.device('cuda:0'),dtype=torch.float)
            out=self.dw(data)
            y.append(out)
            y=torch.cat(y)
        z=sf.img2bas(y,D)
        return z
            

