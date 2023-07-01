import torch
import torch.nn as nn
import torch.nn.functional as F

EPSILON = 1e-10

    
class Convlayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,use_norm, use_relu):
        super(Convlayer, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.use_relu = use_relu
        self.use_norm = use_norm
        
    def forward(self, x):
        x = self.conv2d(x)
        if self.use_norm:
            x =self.bn(x)
        if self.use_relu:
            x = F.relu(x, inplace=False)        
        return x

class CC_attention(nn.Module):
    def __init__(self, kernel_size=[8,1]):
        super(CC_attention, self).__init__()
        self.kernel_size = kernel_size
        self.avg_pooling_H = torch.nn.AvgPool2d((self.kernel_size[0],self.kernel_size[1]), stride=(self.kernel_size[0],self.kernel_size[1]))
        self.avg_pooling_W = torch.nn.AvgPool2d((self.kernel_size[1],self.kernel_size[0]), stride=(self.kernel_size[1],self.kernel_size[0]))
        self.cc_softmax = torch.nn.Softmax(dim=-1)

    def forward(self, tensor1, tensor2):
        # tensor1 base, tensor2 ref
        B, C, H, W = tensor1.size()
        
        query = tensor2
        query_H = query.permute(0, 3, 1, 2).contiguous().view(B, -1, H).permute(0, 2, 1) # [B,H,C*W]
        query_W = query.permute(0, 2 ,1, 3).contiguous().view(B, -1, W).permute(0, 2, 1) # [B,W,C*H]
        
        key_H = self.avg_pooling_H(tensor1) # [B,C,H/8,W]
        key_W = self.avg_pooling_W(tensor1) # [B,C,W/8,H]
        key_size_H = key_H.size()[2]
        key_size_W = key_W.size()[3]
        key_H = key_H.permute(0, 3, 1, 2).contiguous().view(B,-1,key_size_H) # [B,C*W,H/8]
        key_W = key_W.permute(0, 2, 1, 3).contiguous().view(B,-1,key_size_W) # [B,C*W,W/8]

        value_H = self.avg_pooling_H(tensor2) # [B,C,H/8,W]
        value_W = self.avg_pooling_W(tensor2) # [B,C,W/8,H]
        value_size_H = value_H.size()[2]
        value_size_W = value_W.size()[3]
        value_H = value_H.permute(0,3,1,2).contiguous().view(B,-1,value_size_H) # [B,C*W,H/8]
        value_W = value_W.permute(0,2,1,3).contiguous().view(B,-1,value_size_W) # [B,C*W,W/8]

        energy_H = torch.bmm(query_H, key_H).view(B,H*key_size_H)  # [B, H*H/8]
        energy_W = torch.bmm(query_W, key_W).view(B,W*key_size_W)  # [B, W*W/8]
        
        energy = torch.cat([energy_H, energy_W], -1)
        energy_max = torch.max(energy)
        energy_min = torch.min(energy)
        energy =  (energy - energy_min) / (energy_max - energy_min)

        concate = self.cc_softmax(energy)# [B,H*H/8 + W*W/8]
        
        att_H = concate[:,0:H*key_size_H].view(B,H,key_size_H) # [B, H*H/8]
        att_W = concate[:,H*key_size_H:H*key_size_H+W*key_size_W].view(B,W,key_size_W) # [B, W*W/8]
        
        out_H = torch.bmm(value_H, att_H.permute(0, 2, 1)).view(B,W,-1,H).permute(0,2,3,1)
        out_W = torch.bmm(value_W, att_W.permute(0, 2, 1)).view(B,H,-1,W).permute(0,2,1,3)

        # gamma = nn.Parameter(torch.zeros(1))
        gamma = torch.tensor(0.5)

        return gamma*(out_H + out_W) + tensor2

class PixerShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor, use_norm, use_relu):
        super(PixerShuffle, self).__init__()
        self.upscale_factor = upscale_factor
        self.conv = nn.Conv2d(in_channels, out_channels*upscale_factor**2, 3, 1, 1)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor)
        self.use_norm = use_norm
        self.use_relu = use_relu

    def forward(self, x):
        x = self.conv(x)
        x = self.pixelshuffle(x)
        if self.use_norm:
            x =self.bn(x)
        if self.use_relu:
            x = F.relu(x, inplace=False) 
        return x

class net(nn.Module):
    def __init__(self, in_channels, out_channels, upscale_factor, nb_filter=[64,64,64,64]):
        super(net, self).__init__()
        self.nb_filter = nb_filter
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upscale_factor = upscale_factor
        # encoder
        self.conv0 = Convlayer(self.in_channels, nb_filter[0] , kernel_size = 3 ,stride = 1,padding = 1, use_norm=False, use_relu=True) #H*W
        self.en_conv1 = Convlayer(nb_filter[0], nb_filter[0], kernel_size=3, stride=1, padding=1, use_norm=False, use_relu=True) #H*W
        self.en_conv2 = Convlayer(nb_filter[0], nb_filter[1], kernel_size=3, stride=2, padding=1, use_norm=False, use_relu=True) #H/2*W/2
        self.en_conv3 = Convlayer(nb_filter[1], nb_filter[2], kernel_size=3, stride=2, padding=1, use_norm=False, use_relu=True) #H/4*W/4
        self.en_conv4 = Convlayer(nb_filter[2], nb_filter[3], kernel_size=3, stride=2, padding=1, use_norm=False, use_relu=True) #H/8*W/8
        
        # decoder
        self.de_conv1 = PixerShuffle(nb_filter[3]*2, nb_filter[2], upscale_factor, use_norm=False, use_relu=True) #H/4*W/4
        self.de_conv2 = PixerShuffle(nb_filter[2]*2, nb_filter[1], upscale_factor, use_norm=False, use_relu=True) #H/2*W/2
        self.de_conv3 = PixerShuffle(nb_filter[1]*2, nb_filter[0], upscale_factor, use_norm=False, use_relu=True) #H*W
        self.conv1 = Convlayer(nb_filter[0]*2, self.out_channels, kernel_size = 3 ,stride = 1,padding = 1, use_norm=False, use_relu=True)
        
        fusion
        self.cc_attention = CC_attention(kernel_size=[8,1])

        
    def encoder(self, x):
        x = self.conv0(x)
        x1 = self.en_conv1(x)
        x2 = self.en_conv2(x1)
        x3 = self.en_conv3(x2)
        x4 = self.en_conv4(x3)
        return [x1, x2, x3, x4]

    def decoder1(self, x, y):
        [x1, x2, x3, x4] = x
        [y1, y2, y3, y4] = y

        z =self.cc_attention(x4, y4)
        out = self.de_conv1(torch.cat((z, x4), dim=1))

        z = self.cc_attention(out, y3)
        out = self.de_conv2(torch.cat((z, x3), dim=1))
        
        z =self.cc_attention(out, y2)
        out = self.de_conv3(torch.cat((z, x2), dim=1))
        
        z = self.cc_attention(out, y1)
        out = self.conv1(torch.cat((z, x1), dim=1))
        return out

    def decoder2(self, x, y):
        [x1, x2, x3, x4] = x
        [y1, y2, y3, y4] = y

        z =self.cc_attention(x4, y4)
        out = self.de_conv1(torch.cat((z, x4), dim=1))

        z = self.cc_attention(out, y3)
        out = self.de_conv2(torch.cat((z, x3), dim=1))
        
        z =self.cc_attention(out, y2)
        out = self.de_conv3(torch.cat((z, x2), dim=1))
        
        z = self.cc_attention(out, y1)
        out = self.conv1(torch.cat((z, x1), dim=1))
        return out
   
    def forward(self, x, y):
        self.size = y.size()

        x = F.interpolate(x, size=[self.size[-2], self.size[-1]],mode="bicubic")

        x = self.encoder(x)

        y = self.encoder(y)
        
        out = self.decoder1(x,y)
        z = self.encoder(out)
        
        out = self.decoder2(y, z)
        return out

    def pad(self, x1, x2):
        # x2 size padding to x1 size
        shape_x1 = x1.size()
        shape_x2 = x2.size()
        left = 0
        right = 0
        top = 0
        bot = 0
        if shape_x1[3] != shape_x2[3]:
            lef_right = shape_x1[3] - shape_x2[3]
            if lef_right%2 == 0.0:
                left = int(lef_right/2)
                right = int(lef_right/2)
            else:
                left = int(lef_right / 2)
                right = int(lef_right - left)

        if shape_x1[2] != shape_x2[2]:
            top_bot = shape_x1[2] - shape_x2[2]
            if top_bot%2 == 0.0:
                top = int(top_bot/2)
                bot = int(top_bot/2)
            else:
                top = int(top_bot / 2)
                bot = int(top_bot - top)

        reflection_padding = [left, right, top, bot]
        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)
        return x2

    def decoder1_eval(self, x, y):
        [x1, x2, x3, x4] = x
        [y1, y2, y3, y4] = y

        z =self.cc_attention(x4, y4)
        out = self.de_conv1(torch.cat((z, x4), dim=1))

        out = self.pad(y3, out)
        z = self.cc_attention(out, y3)
        out = self.de_conv2(torch.cat((z, x3), dim=1))
        
        out = self.pad(y2, out)
        z =self.cc_attention(out, y2)
        out = self.de_conv3(torch.cat((z, x2), dim=1))
        
        out = self.pad(y1, out)
        z = self.cc_attention(out, y1)
        out = self.conv1(torch.cat((z, x1), dim=1))
        return out

    def decoder2_eval(self, x, y):
        [x1, x2, x3, x4] = x
        [y1, y2, y3, y4] = y

        z =self.cc_attention(x4, y4)
        out = self.de_conv1(torch.cat((z, x4), dim=1))

        out = self.pad(y3, out)
        z = self.cc_attention(out, y3)
        out = self.de_conv2(torch.cat((z, x3), dim=1))
        
        out = self.pad(y2, out)
        z =self.cc_attention(out, y2)
        out = self.de_conv3(torch.cat((z, x2), dim=1))
        
        out = self.pad(y1, out)
        z = self.cc_attention(out, y1)
        out = self.conv1(torch.cat((z, x1), dim=1))
        return out

    def forward_eval(self, x, y):
        self.size = y.size()

        x = F.interpolate(x, size=[self.size[-2], self.size[-1]],mode="bicubic")

        x = self.encoder(x)

        y = self.encoder(y)
        
        out = self.decoder1_eval(x,y)
        z = self.encoder(out)
        
        out = self.decoder2_eval(y, z)
        return out
# net = net(1,1,2)
# print(net)
# test1 = torch.zeros([1,1,65,65])
# test2 = torch.zeros([1,1,255,255])
# a = net.forward_eval(test1, test2)
# print(a.size())
