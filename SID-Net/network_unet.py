"""
model for network
author: Weichao Yi
date: 11/07/21
"""

# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F

# pixelshuffle layer + conv layer for upsample operation
""" 
class Unsample(nn.Module):
    def __init__(self, factor=2,channel=32):
        super(Unsample, self).__init__()
        self.ps = nn.PixelShuffle(factor)
        self.conv = nn.Conv2d(channel//(factor**2),channel,kernel_size=3,padding=1)

    def forward(self, x):
        # ps = nn.PixelShuffle(2)
        y = self.ps(x)
        y = self.conv(y)
        return y
"""


class Unsample(nn.Module):
    def __init__(self, factor=2, in_channel=32, out_channel=32):
        super(Unsample, self).__init__()
        self.ps = nn.UpsamplingBilinear2d(scale_factor=factor)
        self.pw = conv1x1(in_channel, out_channel)
        # self.conv = nn.Conv2d(channel,channel,kernel_size=3,padding=1)
        # self.attention = DP_attention(channel)

    def forward(self, x):
        y = self.ps(x)
        y = self.pw(y)
        # y = self.attention(y)
        return y


class CALayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y




class DP_attention(nn.Module):
    def __init__(self, in_planes):
        super(DP_attention, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes, 3, 1, 1, groups=in_planes, bias=False)  # dw_conv
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_planes, in_planes, 1, bias=False)  # pw_conv
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.conv2(out)
        weight = self.sigmoid(out)
        out = out * weight
        return out


# for channel change
def conv1x1(in_planes, out_planes, stride=1):
    "1x1 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)


# --- Main model  --- #

class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1

# class Recurrent_block(nn.Module):
#     def __init__(self, ch_out, ):
#         super(Recurrent_block, self).__init__()
#         self.ch_out = ch_out
#         self.conv = nn.Sequential(
#             nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True)
#         )
#
#     def forward(self, x):
#         x1 = self.conv(x)
#         return x1



class Fusion(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Fusion,self).__init__()
        self.up_sample = Unsample(in_channel=in_channel,out_channel=out_channel)
        self.softmax = nn.Sigmoid()
        self.conv = nn.Conv2d(2*out_channel,out_channel,1)

    def forward(self, f_l,f_h):
        f_i = torch.cat([f_l,self.up_sample(f_h)],dim=1)
        weight = self.softmax(self.conv(f_i))
        f_o = torch.add(weight*f_l,(1-weight)*self.up_sample(f_h))
        return f_o







class main_Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(main_Block, self).__init__()
        self.dilaconv1 = Recurrent_block(in_channel)
        # self.dilaconv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        # self.dilaconv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.dilaconv2 = Recurrent_block(out_channel)
        # self.dilaconv2 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        # self.dilaconv3 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=3, dilation=3)
        # self.dilaconv5 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=5, dilation=5)
        # self.pwconv = conv1x1(2 * in_channel, out_channel)
        self.relu = nn.ReLU()
        self.attention1 = CALayer(out_channel)

    def forward(self, x):
        x0 = x
        y_1 = self.relu(self.dilaconv1(x))
        # y_2 = self.attention1(self.dilaconv2(y_1))
        y_2 = self.dilaconv2(y_1)
        # y = self.pwconv(torch.cat([y_1_3, y_3_5], dim=1))
        y = self.attention1(y_2)
        y = torch.add(y, x)
        return y



"""
class Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Block, self).__init__()
        self.dilaconv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        # self.dilaconv2 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.dilaconv3 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=3, dilation=3)
        self.dilaconv5 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=5, dilation=5)
        self.pwconv = conv1x1(2 * in_channel, out_channel)
        self.attention = DP_attention(out_channel)
        self.relu = nn.ReLU()
        # self.attention1 = CALayer(out_channel)

    def forward(self, x):
        x0 = x
        y_1_3 = self.relu(self.dilaconv1(x) + self.dilaconv3(x))
        y_3_5 = self.relu(self.dilaconv3(x) + self.dilaconv5(x))
        y = self.pwconv(torch.cat([y_1_3, y_3_5], dim=1))
        y = self.attention(y)
        y = y + x0
        return y
"""

"""
class Group(nn.Module):
    def __init__(self, input_size, output_size, num=3):
        super(Group, self).__init__()
        modules = [Block(input_size, output_size) for _ in range(num)]
        modules.append(nn.Conv2d(output_size, output_size, kernel_size=3, padding=1))
        self.gp = nn.Sequential(*modules)

    def forward(self, x):
        res = self.gp(x)
        res = res + x
        return res
"""


class main_Group(nn.Module):
    def __init__(self, input_size, output_size, num=3):
        super(main_Group, self).__init__()

        self.block1 = main_Block(input_size, output_size)
        self.block2 = main_Block(output_size, output_size)
        # self.block3 = main_Block(output_size, output_size)
        # self.block4 = Block(input_size,output_size)
        self.conv = nn.Conv2d(output_size, output_size, kernel_size=3, padding=1)
        # self.pw_conv1 = conv1x1(2*output_size,output_size)
        # self.pw_conv2 = conv1x1(3*output_size,output_size)
        # modules = [Block(input_size, output_size) for _ in range(num)]
        # modules.append(nn.Conv2d(output_size, output_size, kernel_size=3, padding=1))
        # self.gp = nn.Sequential(*modules)

    def forward(self, x):
        y1 = self.block1(x)
        y2 = self.conv(y1)
        y3 = self.block2(y2)
        # y3 = self.block3(y2)

        # y = y4+x
        y = torch.add(y3, x)
        # res = self.gp(x)
        # res = res + x
        return y



class Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Block, self).__init__()
        self.dilaconv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.dilaconv3 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=3, dilation=3)
        self.dilaconv5 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=5, dilation=5)
        self.pw_conv1 = conv1x1(2 * out_channel, out_channel)
        self.pw_conv2 = conv1x1(2 * out_channel, out_channel)
        self.l_relu = nn.LeakyReLU()
        self.conv = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        y1_1 = self.dilaconv1(x)
        y1_3 = self.dilaconv3(x)
        y1_5 = self.dilaconv5(x)
        y1 = self.pw_conv1(torch.cat([y1_1, y1_3], dim=1))
        y2 = self.l_relu(self.pw_conv2(torch.cat([y1, y1_5], dim=1)))
        y3 = self.conv(y2)
        y4 = torch.add(y3, x)
        return y4


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.sub1_layer1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.sub1_group1 = main_Group(32, 32)

        self.dp_layer1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.sub1_group2 = main_Group(64, 64)
        self.fusion1 = Fusion(64, 32)

        self.dp_layer2 = nn.Conv2d(64, 96, kernel_size=3, stride=2, padding=1)

        self.fusion2 = Fusion(96, 64)
        self.sub1_group3 = main_Group(96, 96)
        self.sub1_group4 = main_Group(96, 96)

        self.up_layer1 = Unsample(2, 96, 64)

        self.sub1_group5 = main_Group(64, 64)

        self.up_layer2 = Unsample(2, 64, 32)

        self.sub1_group6 = main_Group(32, 32)

        self.out_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, hazy):  # 1/4,1/2,1
        stage1_1 = self.sub1_group1(self.relu(self.sub1_layer1(hazy)))  # 32 16 16
        # stage2_1 = self.relu(self.sub2_layer1(hazy1))

        stage1_2 = self.sub1_group2(self.dp_layer1(stage1_1))  # 64 8 8
        # stage2_2 = self.sub2_group1(stage2_1)
        fusion_feature_0 = self.fusion1(stage1_1,stage1_2) # 32 16 16



        # stage1_3 = self.sub1_group4(self.sub1_group3(self.dp_layer2(stage1_2)))  # 96 4 4


        stage1_3 = self.sub1_group3(self.dp_layer2(stage1_2))  # 96 4 4
        fusion_feature_1 = self.fusion2(stage1_2,stage1_3) # 64 8 8

        stage1_3_1 = self.sub1_group4(stage1_3)
        # stage2_3 = self.sub2_group2(torch.add(stage1_2,stage2_2))
        # stage1_4 = torch.add(self.sub1_group5(self.up_layer1(stage1_3_1)), stage1_2)  # 64 8 8
        stage1_4 = torch.add(self.sub1_group5(self.up_layer1(stage1_3_1)), fusion_feature_1)  # 64 8 8
        # stage1_4 = self.sub1_group5(self.up_layer1(stage1_3_1)) # 64 8 8
        # stage1_5 = self.sub1_group6(self.up_layer2(stage1_4)) # 32 16 16
        stage1_5 = torch.add(self.sub1_group6(self.up_layer2(stage1_4)), fusion_feature_0)  # 32 16 16
        # stage1_5 = torch.add(self.sub1_group6(self.up_layer2(stage1_4)), stage1_1)  # 32 16 16

        # stage1_4 = self.sub1_group3(torch.add(stage1_3,stage2_3))
        # stage2_4 = self.sub2_group3(torch.add(stage1_3,stage2_3))

        # fusion = torch.cat([stage1_4,stage2_4],dim=1)
        # y = self.pw_conv(self.attention(fusion))
        clean = self.tanh(self.out_conv(stage1_5))
        clean = torch.add(clean, hazy)
        # return y1
        # return clean1,clean2,clean
        return clean


if __name__ == "__main__":
    N, C_in, H, W = 1, 3, 16, 16
    x = torch.randn(N, C_in, H, W).float()
    x1 = torch.randn(N, C_in, H // 2, W // 2).float()
    # x2 = torch.randn(N, C_in, H // 4, W // 4).float()
    # y = Fusion(3,3)
    y = Net()
    print(y)
    # _,_,result = y(x2,x1,x)
    result = y(x)
    # print(result)
    print(result.shape)
    print("groups=in_channels时参数大小：%d" % sum(param.numel() for param in y.parameters()))
