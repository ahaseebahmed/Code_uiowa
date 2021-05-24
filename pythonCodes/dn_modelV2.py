
import miscTorch as sf
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


class SmallModel(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.deconv00 = nn.ConvTranspose2d(1,d*16,(1,1),(1,1),(0,0),bias=False)
#        self.deconv0 = nn.ConvTranspose2d(d*16,d*16,(3,3),(1,1),(0,1),bias=False)
#        self.deconv1 = nn.ConvTranspose2d(d*16,d*8,(3,3),(1,1),(0,1),bias=False)
#        self.deconv2 = nn.ConvTranspose2d(d*8,d*4,(4,3),(2,1),(1,1),bias=False)
        #self.deconv2 = nn.ConvTranspose2d(d*8,d*4,(4,3),(2,1),(1,1),bias=False)
        #self.deconv3 = nn.ConvTranspose2d(d*4,d*4,(4,3),(2,1),(2,1))
        #self.deconv4 = nn.ConvTranspose2d(1,1,(4,3),(2,1),(1,1))
        self.deconv0 = nn.ConvTranspose2d(1,d*16,(4,3),(1,1),(0,1),bias=False)
        #self.deconv1 = nn.ConvTranspose2d(d*16,d*12,(1,1),(1,1),(0,0),bias=False)
        self.deconv2 = nn.ConvTranspose2d(d*16,d*12,(4,3),(1,1),(0,1),bias=False)
        #self.deconv3 = nn.ConvTranspose2d(d*12,d*8,(1,1),(1,1),(0,0),bias=False)
        self.deconv4 = nn.ConvTranspose2d(d*12,d*8,(4,3),(1,1),(0,1),bias=False)
        #self.deconv5 = nn.ConvTranspose2d(d*8,d*4,(1,1),(1,1),(0,0),bias=False)
        self.deconv6 = nn.ConvTranspose2d(d*8,d*4,(4,3),(1,1),(0,1),bias=False)
        self.deconv7 = nn.ConvTranspose2d(d*4,d*2,(4,3),(1,1),(0,1),bias=False)
        self.deconv8 = nn.ConvTranspose2d(d*2,d,(4,3),(1,1),(0,1),bias=False)
        self.lay1=layer(d,d*4)
        #self.lay2=layer(d*8,d*4)
        self.lay3=layer(d*4,d*4)
#        #self.lay4=layer(256,512)
#        #self.lay5=layer(512,512)
#        #self.lay6=layer(512,512)
#        #self.lay7=layer(512,256)
#        self.lay8=layer(256,128)
#        self.lay9=layer(128,64)
        self.lay10=nn.Conv2d(d*4,1,3,1,padding=1,bias=False)


    def forward(self,inp):
        #inp,mn,st=sf.pt_normalizeMeanStd(inp)
        #x = F.leaky_relu(self.deconv00(inp),0)
        x = F.leaky_relu(self.deconv0(inp),0)
        #x = F.leaky_relu(self.deconv1(x),0)
        x = F.leaky_relu(self.deconv2(x),0)
        #x = F.leaky_relu(self.deconv3(x),0)
        x = F.leaky_relu(self.deconv4(x),0)
        #x = F.leaky_relu(self.deconv5(x),0)
        x = F.leaky_relu(self.deconv6(x),0)
        x = F.leaky_relu(self.deconv7(x),0)
        x = F.leaky_relu(self.deconv8(x),0)

        #x = F.relu(self.deconv3(x))
        #x = F.relu(self.deconv4(x))
        x=self.lay1(x)
        #x=self.lay2(x)
        x=self.lay3(x)
#        #x=self.lay4(x)
#        #x=self.lay5(x)
#        #x=self.lay6(x)
#        #x=self.lay7(x)
#        x=self.lay8(x)
#        x=self.lay9(x)
        x=self.lay10(x)
        #x1=x+inp
        #x1=x1*st+mn
        
        return x
