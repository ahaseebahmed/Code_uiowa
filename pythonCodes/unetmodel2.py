"""
This is the unet model in pytorch
@author: ahhmed
"""
import miscTorch as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

class layer(nn.Module):
    def __init__(self,ich,och):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=ich,out_channels=och,kernel_size=3,
                             padding=1,bias=False)
        self.bn=nn.BatchNorm2d(och)
    def forward(self,x):
        x=self.conv1(x)
        #x=self.bn(x)
        x=F.relu(x)
        return x

class UnetClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.lay1=layer(60,128)
        self.lay2=layer(128,128)

        self.lay3=layer(128,256)
        self.lay4=layer(256,256)

        self.lay5=layer(256,512)
        self.lay6=layer(512,512)

        self.lay7=layer(512,1024)
        self.lay8=layer(1024,1024)
        
        self.lay9=layer(1024,2048)
        self.lay10=layer(2048,2048)
        
        self.lay11=layer(2048,4096)
        self.lay12=layer(4096,4096)

        self.lay13=layer(4096,4096)

        self.lay14=layer(8192,2048)
        self.lay15=layer(2048,2048)

        self.lay16=layer(4096,1024)
        self.lay17=layer(1024,1024)

        self.lay18=layer(2048,512)
        self.lay19=layer(512,512)

        self.lay20=layer(1024,256)
        self.lay21=layer(256,256)
        
        self.lay22=layer(512,128)
        self.lay23=layer(128,128)
        
        self.lay24=layer(256,64)
        self.lay25=layer(64,64)

        self.lay26=nn.Conv2d(64,60,3,1,padding=1,bias=False)

        #self.pool=nn.AvgPool2d(2,2)
        #self.unpool=lambda x,sz: F.interpolate(x,size=sz,mode='bilinear',
        #                                  align_corners=False)
        #self.unpool=F.interpolate(input)

    def forward(self, inp):

        #inp,mn,st=sf.pt_normalizeMeanStd(inp)
        x=self.lay1(inp)
        x1=self.lay2(x)
        p1=F.avg_pool2d(x1,(2,2))

        x=self.lay3(p1)
        x2=self.lay4(x)
        p2=F.avg_pool2d(x2,(2,2))


        x=self.lay5(p2)
        x3=self.lay6(x)
        p3=F.avg_pool2d(x3,(2,2))


        x=self.lay7(p3)
        x4=self.lay8(x)
        p4=F.avg_pool2d(x4,(2,2))

        x=self.lay9(p4)
        x5=self.lay10(x)
        p5=F.avg_pool2d(x5,(2,2))
        
        x=self.lay11(p5)
        x6=self.lay12(x)
        p6=F.avg_pool2d(x6,(2,2))

        x=self.lay13(p6)


        x=F.interpolate(x,x6.shape[-2:],mode='bilinear',align_corners=False)
        x=torch.cat([x,x6],dim=-3)
        x=self.lay14(x)
        x=self.lay15(x)
        
        
        x=F.interpolate(x,x5.shape[-2:],mode='bilinear',align_corners=False)
        x=torch.cat([x,x5],dim=-3)
        x=self.lay16(x)
        x=self.lay17(x)
        
        x=F.interpolate(x,x4.shape[-2:],mode='bilinear',align_corners=False)
        x=torch.cat([x,x4],dim=-3)
        x=self.lay18(x)
        x=self.lay19(x)

        x=F.interpolate(x,x3.shape[-2:],mode='bilinear',align_corners=False)
        x=torch.cat([x,x3],dim=-3)
        x=self.lay20(x)
        x=self.lay21(x)

        x=F.interpolate(x,x2.shape[-2:],mode='bilinear',align_corners=False)
        x=torch.cat([x,x2],dim=-3)
        x=self.lay22(x)
        x=self.lay23(x)


        x=F.interpolate(x,x1.shape[-2:],mode='bilinear',align_corners=False)
        x=torch.cat([x,x1],dim=-3)
        x=self.lay24(x)
        x=self.lay25(x)

        x=self.lay26(x)

        #x=x+inp
        #x=x*st+mn

        return x

#%%

# unet=UnetClass()
# x=torch.randn(256,232,dtype=torch.float32)
# x=x[None,None]
# y=unet(x)

