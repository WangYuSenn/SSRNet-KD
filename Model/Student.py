import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.P2T.p2t import p2t_tiny
from config_tld import cfg
from collections import OrderedDict
class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel, padding, dilation):
        super(DepthWiseConv, self).__init__()
        self.in_channels = in_channel
        self.out_channels = out_channel
        self.depth_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=False,
                                    groups=in_channel),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True)
        )
        self.point_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    bias=False,
                                    groups=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, bias = False):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding,bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class merge_out(nn.Module):
    def __init__(self, in1,in2):
        super(merge_out, self).__init__()
        self.last_conv1 = BasicConv2d(in2, in1, kernel_size=1, stride=1, padding=0)
        self.last_conv2 = BasicConv2d(in1*2, in1, kernel_size=1, stride=1, padding=0)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x, y):
        y = self.last_conv1(y)
        y = self.up(y)
        x = torch.cat((x,y),dim=1)
        out = self.last_conv2(x)
        return out

class WAM(nn.Module):
    def __init__(self, in_chanel):
        super(WAM, self).__init__()
        self.in_channel = in_chanel
        self.linearW = nn.Linear(in_chanel, in_chanel, bias=False)
    def forward(self, x1):
        size = x1.size()[2:]
        all_dim = size[0] * size[1]
        x1 = x1.view(-1, self.in_channel, all_dim)
        x11 = torch.transpose(x1, 1, 2).contiguous()
        x1_corr = self.linearW(x11)
        x111 = torch.bmm(x1, x1_corr)
        a1 = F.softmax(x111.clone(), dim=2)
        a1 = F.softmax(a1, dim=1)
        x1_out = torch.bmm(a1, x1).contiguous()
        x1_out = x1_out + x1
        out = x1_out.view(-1, self.in_channel, size[0], size[1])
        return out
class SRM(nn.Module):
    def __init__(self,in_planes, out_planes, reduction=16, bn_momentum=0.0003):
        super(SRM, self).__init__()
        self.in_planes = in_planes
        self.bn_momentum = bn_momentum
        self.adm = WAM(in_planes)
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        channel_weight = self.adm(x)
        x_x = channel_weight * x
        atten_v = self.softmax(x_x)
        merge_out = x * atten_v + atten_v
        return merge_out
class DE(nn.Module):
    def __init__(self, in_channel, out_channel):
        depth = in_channel
        super(DE, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d(1)
        self.conv = DepthWiseConv(in_channel, depth, 1, 1)
        self.atrous_block1 = DepthWiseConv(in_channel, depth, 1, 1)
        self.atrous_block6 = DepthWiseConv(in_channel, depth, padding=4, dilation=4)
        self.atrous_block12 = DepthWiseConv(in_channel, depth, padding=6, dilation=6)
        self.atrous_block18 = DepthWiseConv(in_channel, depth, padding=8, dilation=8)
        self.conv_1x1_output = BasicConv2d(depth * 5, out_channel, 1, 1, 0)
    def forward(self, x):
        size = x.shape[2:]
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.interpolate(image_features, size=size, mode='bilinear', align_corners=True)
        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)
        cat = torch.cat([image_features, atrous_block1, atrous_block6,
                         atrous_block12, atrous_block18], dim=1)
        net = self.conv_1x1_output(cat)
        return net
class SpatialAttention(nn.Module):
    def __init__(self, channel):
        super(SpatialAttention, self).__init__()

        self.convs = nn.Sequential(
            nn.Conv2d(channel, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.convs(x)
        return x
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=2):
        super(ChannelAttention, self).__init__()
        squeeze_channels = max(in_planes // ratio, 16)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, squeeze_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(squeeze_channels, in_planes, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.fc(F.adaptive_avg_pool2d(x, 1)) * x
        return out
class MScoreCa(nn.Module):
    def __init__(self, channel):
        super(MScoreCa,self).__init__()
        self.sa = SpatialAttention(channel)
        self.ca = ChannelAttention(channel)
    def forward(self, f):
        bs, c, h, w = f.shape
        M_S = f.view(bs,-1)
        v = torch.var(M_S, dim=0)
        v = torch.sigmoid(v)
        v_co = v.unsqueeze(0)
        v_copy = v_co.repeat(bs, 1)
        feature_sorce = v_copy.reshape(bs,c,h,w)
        feature_sorce = torch.sigmoid(feature_sorce)
        out = self.ca(feature_sorce)
        return out
class MScoreSa(nn.Module):
    def __init__(self, channel):
        super(MScoreSa,self).__init__()
        self.sa = SpatialAttention(channel)
        self.ca = ChannelAttention(channel)
    def forward(self, f):
        bs, c, h, w = f.shape
        M_S = f.view(bs,-1)
        v = torch.var(M_S, dim=0)
        v = torch.sigmoid(v)
        v_co = v.unsqueeze(0)
        v_copy = v_co.repeat(bs, 1)
        feature_sorce = v_copy.reshape(bs,c,h,w)
        feature_sorce = torch.sigmoid(feature_sorce)
        out = self.sa(feature_sorce)
        return out
class Student(nn.Module):
    def __init__(self, config=cfg):
        super(Student, self).__init__()
        self.backbone = p2t_tiny()
        path = "/home/xug/PycharmProjects/TLD/backbone/P2T/retinanet_p2t_t_fpn_1x_coco-1e0959bd.pth"
        sk = torch.load(path)['state_dict']
        new_state_dice = OrderedDict()
        for k, v in sk.items():
            name = k[9:]
            new_state_dice[name] = v
        self.backbone.load_state_dict(new_state_dice, strict=False)
        self.cfg = config
        self.ms3 = MScoreCa(640)
        self.ms2 = MScoreCa(320)
        self.ms1 = MScoreSa(128)
        self.ms0 = MScoreSa(64)
        self.pre1 = BasicConv2d(48, 64, 3, 1, 1)
        self.aspp00 = DE(64,64)
        self.pre2 = BasicConv2d(96, 128, 3, 1, 1)
        self.DE3 = DE(240, 320)
        self.DE4 = DE(384, 640)
        self.csc0 = SRM(64, 64)
        self.csc1 = SRM(128, 128)
        self.csc2 = SRM(320, 320)
        self.csc3 = SRM(640, 640)
        self.merge1 = merge_out(320, 640)
        self.merge2 = merge_out(128, 320)
        self.merge3 = merge_out(64, 128)
        self.conv64_3 = BasicConv2d(64, 1, 3, 1, 1)
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv128to640 = BasicConv2d(128, 640, 1, 1, 0)
        self.conv64to1 = BasicConv2d(64, 1, 1, 1, 0)
    def forward(self, x):
        x1_rgb, x2_rgb, x3_rgb, x4_rgb = self.backbone(x)
        x1_rgb = self.pre1(x1_rgb)
        x2_rgb = self.pre2(x2_rgb)
        x3_rgb = self.DE3(x3_rgb)
        x4_rgb = self.DE4(x4_rgb)
        csc0 = self.csc0(x1_rgb)
        csc1 = self.csc1(x2_rgb)
        csc2 = self.csc2(x3_rgb)
        csc3 = self.csc3(x4_rgb)
        ms3 = self.ms3(csc3)
        ms2 = self.ms2(csc2)
        ms1 = self.ms1(csc1)
        xx = self.aspp00(csc0)
        ms0 = self.ms0(xx)
        merge_1 = self.merge1(csc2,csc3)
        merge_2 = self.merge2(csc1,merge_1)
        merge_3 = self.merge3(csc0,merge_2)
        out = self.conv64_3(self.up(merge_3))
        csc3_use = F.interpolate(csc1, size = csc3.size()[2:], mode = 'bilinear', align_corners=True)
        csc3_use = self.conv128to640(csc3_use)
        x1_rgb_use = F.interpolate(x1_rgb, size = out.size()[2:], mode = 'bilinear', align_corners=True)
        x1_rgb_use = self.conv64to1(x1_rgb_use)
        return out, csc0, csc1, csc2, csc3, csc3_use, ms3, ms2, ms1, ms0, x1_rgb_use
