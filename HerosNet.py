import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np
from Utils import *

class MySign(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = input.new(input.size())
        output[input >= 0] = 1.
        output[input < 0] = 0.
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input

MyBinarize = MySign.apply

class Resblock(nn.Module):
    def __init__(self, HBW):
        super(Resblock, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1))
        self.block2 = nn.Sequential(nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1),
                                    nn.ReLU(),
                                    nn.Conv2d(HBW, HBW, kernel_size=3, stride=1, padding=1))

    def forward(self, x):
        tem = x
        r1 = self.block1(x)
        out = r1 + tem
        r2 = self.block2(out)
        out = r2 + out
        return out

## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=4, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return y

## Supervised Attention Module for Cross-Phase interaction in HFIM##
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size//2), bias=bias, stride=1)
        self.conv2 = nn.Conv2d(n_feat, 28, kernel_size, padding=(kernel_size//2), bias=bias, stride=1)
        self.conv3 = nn.Conv2d(28, n_feat, kernel_size, padding=(kernel_size//2), bias=bias, stride=1)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img

## Res-Encoder
class Encoding(nn.Module):
    def __init__(self):
        super(Encoding, self).__init__()
        self.E1 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.E2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.E3 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.E4 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.E5 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        
        Res_list = [Resblock(128) for _ in range(16)]
        self.Res = nn.Sequential(*Res_list)

    def forward(self, x):
        ## encoding blocks
        E1 = self.E1(x)
        E2 = self.E2(F.avg_pool2d(E1, kernel_size=2, stride=2))
        E3 = self.E3(F.avg_pool2d(E2, kernel_size=2, stride=2))
        E4 = self.E4(F.avg_pool2d(E3, kernel_size=2, stride=2))
        E5 = self.E5(F.avg_pool2d(E4, kernel_size=2, stride=2))
        E5 = self.Res(E5)

        return E1, E2, E3, E4, E5

class Decoding(nn.Module):
    def __init__(self, Ch=28):
        super(Decoding, self).__init__()
        self.upMode = 'bilinear'
        self.Ch = Ch
        self.D1 = nn.Sequential(nn.Conv2d(in_channels=128+128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.D2 = nn.Sequential(nn.Conv2d(in_channels=128+64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.D3 = nn.Sequential(nn.Conv2d(in_channels=64+64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                                nn.ReLU()
                                )
        self.D4 = nn.Sequential(nn.Conv2d(in_channels=64+32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(),
                                nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                                )


    def forward(self, E1, E2, E3, E4, E5):
        ## decoding blocks
        D1 = self.D1(torch.cat([E4, F.interpolate(E5, scale_factor=2, mode=self.upMode)], dim=1))
        D2 = self.D2(torch.cat([E3, F.interpolate(D1, scale_factor=2, mode=self.upMode)], dim=1))
        D3 = self.D3(torch.cat([E2, F.interpolate(D2, scale_factor=2, mode=self.upMode)], dim=1))
        D4 = self.D4(torch.cat([E1, F.interpolate(D3, scale_factor=2, mode=self.upMode)], dim=1))

        return D4


class HerosNet(nn.Module):
    def __init__(self, Ch, stages, size):
        super(HerosNet, self).__init__()
        self.Ch = Ch
        self.s  = stages
        self.size = size

        ## Mask Initialization ##
        self.Phi = Parameter(torch.ones(self.size, self.size), requires_grad=True)
        torch.nn.init.normal_(self.Phi, mean=0, std=0.1)
        
        ## DGDM ##

        ## The modules for simulating the measurement matrix A and A^T
        self.AT = nn.Sequential(nn.Conv2d(Ch, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                Resblock(64), Resblock(64),
                                nn.Conv2d(64, Ch, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())
        self.A  = nn.Sequential(nn.Conv2d(Ch, 64, kernel_size=3, stride=1, padding=1), nn.LeakyReLU(),
                                Resblock(64), Resblock(64),
                                nn.Conv2d(64, Ch, kernel_size=3, stride=1, padding=1), nn.LeakyReLU())
        
        ## Static component of the dynamic step size mechanism ##
        self.delta_0 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_1 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_2 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_3 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_4 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_5 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_6 = Parameter(torch.ones(1), requires_grad=True)
        self.delta_7 = Parameter(torch.ones(1), requires_grad=True)
        
        torch.nn.init.normal_(self.delta_0, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_1, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_2, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_3, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_4, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_5, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_6, mean=0.1, std=0.01)
        torch.nn.init.normal_(self.delta_7, mean=0.1, std=0.01)
        
        ## Dynamic component of the dynamic step size mechanism ##
        self.catt = CALayer(channel=28, reduction=4, bias=False)
        self.cons = Parameter(torch.Tensor([0.5]))
       
        ## Enhancement_Module ##
        self.Encoding = Encoding()
        self.Decoding = Decoding(Ch=self.Ch)
         
        ## Cross-phase Fusion in the HFIM ##
        self.conv  = nn.Conv2d(Ch, 32, kernel_size=3, stride=1, padding=1)

        self.conv_sam1 = nn.Conv2d(32 * 2, 32, kernel_size=1, stride=1, padding=0)
        self.conv_sam2 = nn.Conv2d(32 * 3, 32, kernel_size=1, stride=1, padding=0)
        self.conv_sam3 = nn.Conv2d(32 * 4, 32, kernel_size=1, stride=1, padding=0)
        self.conv_sam4 = nn.Conv2d(32 * 5, 32, kernel_size=1, stride=1, padding=0)
        self.conv_sam5 = nn.Conv2d(32 * 6, 32, kernel_size=1, stride=1, padding=0)
        self.conv_sam6 = nn.Conv2d(32 * 7, 32, kernel_size=1, stride=1, padding=0)
        self.conv_sam7 = nn.Conv2d(32 * 8, 32, kernel_size=1, stride=1, padding=0)

        ## Cross-phase Interaction in the HFIM ##
        self.SAM0 = SAM(n_feat=32, kernel_size=3, bias=False)
        self.SAM1 = SAM(n_feat=32, kernel_size=3, bias=False)
        self.SAM2 = SAM(n_feat=32, kernel_size=3, bias=False)
        self.SAM3 = SAM(n_feat=32, kernel_size=3, bias=False)
        self.SAM4 = SAM(n_feat=32, kernel_size=3, bias=False)
        self.SAM5 = SAM(n_feat=32, kernel_size=3, bias=False)
        self.SAM6 = SAM(n_feat=32, kernel_size=3, bias=False)
        self.SAM7 = SAM(n_feat=32, kernel_size=3, bias=False)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data)
                #nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.constant_(m.bias.data, 0.0)

    def recon(self, res1, Xt, i):
        if i == 0 :
            delta = self.delta_0
        elif i == 1:
            delta = self.delta_1
        elif i == 2:
            delta = self.delta_2
        elif i == 3:
            delta = self.delta_3
        elif i == 4:
            delta = self.delta_4
        elif i == 5:
            delta = self.delta_5
        elif i == 6:
            delta = self.delta_6
        elif i == 7:
            delta = self.delta_7

        Xt     =   Xt - delta * res1 - self.cons * self.catt(Xt) * res1
        return Xt

    def forward(self, training_label):
        
        ## Sampling Subnet ##
        batch, _, _, _ = training_label.shape
        Phi_ = MyBinarize(self.Phi)

        PhiWeight = Phi_.contiguous().view(1,1,self.size,self.size)
        PhiWeight = PhiWeight.repeat(batch,28,1,1)

        temp = training_label.mul(PhiWeight)
        temp_shift = torch.Tensor(np.zeros((batch, 28, self.size, self.size + (28 - 1) * 2))).cuda()
        temp_shift[:,:,:, 0:self.size] = temp
        for t in range(28):
            temp_shift[:,t,:,:] = torch.roll(temp_shift[:,t,:,:], 2 * t, dims=2)
        meas = torch.sum(temp_shift, dim=1).cuda()

        y = meas / 28 * 2
        y = y.unsqueeze(1).cuda()

        ## Initialization Subnet ##
        Xt = y2x(y)
        Xt_ori = Xt

        OUT = []
        HT = []
     
        ## Recovery Subnet ##
        for i in range(0, self.s):
            ## DGDM ##
            AXt = x2y(self.A(Xt))
            Res1 = self.AT(y2x(AXt - y))

            Xt = self.recon(Res1, Xt, i)
            
            ## convert Xt from the image domain to the feature domain ##
            fea = self.conv(Xt)

            ## Cross-phase Fusion in HFIM ##
            if i == 0:
                pass
            elif i == 1:
                HT.append(fea)
                fea = self.conv_sam1(torch.cat(HT, 1))
                HT.pop()
            elif i == 2:
                HT.append(fea)
                fea = self.conv_sam2(torch.cat(HT, 1))
                HT.pop()
            elif i == 3:
                HT.append(fea)
                fea = self.conv_sam3(torch.cat(HT, 1))
                HT.pop()
            elif i == 4:
                HT.append(fea)
                fea = self.conv_sam4(torch.cat(HT, 1))
                HT.pop()
            elif i == 5:
                HT.append(fea)
                fea = self.conv_sam5(torch.cat(HT, 1))
                HT.pop()
            elif i == 6:
                HT.append(fea)
                fea = self.conv_sam6(torch.cat(HT, 1))
                HT.pop()
            elif i == 7:
                HT.append(fea)
                fea = self.conv_sam7(torch.cat(HT, 1))
                HT.pop()

            ## Enhancement Module ##
            E1, E2, E3, E4, E5 = self.Encoding(fea)
            Xt = self.Decoding(E1, E2, E3, E4, E5)
            
            ## Cross-phase Interaction in HFIM ##
            if i == 0:
                Ht, Xt = self.SAM0(Xt, Xt_ori)
            elif i == 1:
                Ht, Xt = self.SAM1(Xt, Xt_ori)
            elif i == 2:
                Ht, Xt = self.SAM2(Xt, Xt_ori)
            elif i == 3:
                Ht, Xt = self.SAM3(Xt, Xt_ori)
            elif i == 4:
                Ht, Xt = self.SAM4(Xt, Xt_ori)
            elif i == 5:
                Ht, Xt = self.SAM5(Xt, Xt_ori)
            elif i == 6:
                Ht, Xt = self.SAM6(Xt, Xt_ori)
            elif i == 7:
                Ht, Xt = self.SAM7(Xt, Xt_ori)

            OUT.append(Xt)
            HT.append(Ht)

        return OUT, Phi_
