import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone.P2T.p2t import p2t_large
from config_tld import cfg
from collections import OrderedDict
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, bias =False):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, bias = bias)
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
        self.last_conv3 = BasicConv2d(in1, in1, kernel_size=3, stride=1, padding=1)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x, y):
        y = self.last_conv1(y)
        y = self.up(y)
        x = torch.cat((x,y),dim=1)
        out = self.last_conv2(x)
        return out
class MLPFilterLayer(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(MLPFilterLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, out_planes // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(out_planes // reduction, out_planes),
            nn.Sigmoid()
        )
        self.out_planes = out_planes
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, self.out_planes, 1, 1)
        return y
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
class MScoreCA(nn.Module):
    def __init__(self, channel):
        super(MScoreCA,self).__init__()
        self.sa = SpatialAttention(channel)
        self.ca = ChannelAttention(channel)
    def forward(self, f):
        bs, c, h, w = f.shape
        M_S = f.view(bs,-1)
        v = torch.var(M_S, dim=0)
        v = torch.sigmoid(v)
        v_co = v.unsqueeze(0)
        v_copy = v_co.repeat(bs,1)
        feature_sorce = v_copy.reshape(bs, c, h, w)
        feature_sorce = torch.sigmoid(feature_sorce)
        out = self.ca(feature_sorce)
        return out
class MScoreSA(nn.Module):
    def __init__(self, channel):
        super(MScoreSA,self).__init__()
        self.sa = SpatialAttention(channel)
        self.ca = ChannelAttention(channel)
    def forward(self, f):
        bs, c, h, w = f.shape
        bc = bs*c
        M_S = f.view(bs,-1)
        v = torch.var(M_S, dim=0)
        v = torch.sigmoid(v)
        v_co = v.unsqueeze(0)
        v_copy = v_co.repeat(bs,1)
        feature_sorce = v_copy.reshape(bs, c, h, w)
        feature_sorce = torch.sigmoid(feature_sorce)
        out = self.sa(feature_sorce)
        return out
'''
Feature Separation Part
'''
class CFP(nn.Module):
    def __init__(self, in_planes, out_planes, reduction=16):
        super(CFP, self).__init__()
        self.filter = MLPFilterLayer(in_planes, out_planes, reduction)
        self.wam = WAM(in_planes)
        self.conv1x1 = BasicConv2d(2 * in_planes, out_planes, 1, 1, 0)
        self.conv3x3 = BasicConv2d(in_planes, out_planes, 3, 1, 1)
    def forward(self, guidePath, mainPath):
        combined = torch.cat((guidePath, mainPath), dim=1)
        combined = self.conv1x1(combined)
        channel_weight = self.wam(combined)
        channel_weight = self.conv3x3(channel_weight)
        channel_weight = self.filter(channel_weight)
        out = guidePath * channel_weight + mainPath
        return out
class SSRM(nn.Module):
    def __init__(self,in_planes, out_planes,reduction=16, bn_momentum=0.0003):
        super(SSRM, self).__init__()
        self.in_planes = in_planes
        self.cfp_rgb = CFP(in_planes, out_planes, reduction)
        self.cfp_t = CFP(in_planes, out_planes, reduction)
        self.gate_rgb = BasicConv2d(in_planes * 2, 1, 1, 1, 0, bias=True)
        self.gate_t = BasicConv2d(in_planes * 2, 1, 1, 1, 0, bias=True)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x, t):
        re_x = self.cfp_rgb(t, x)
        re_t = self.cfp_t(x, t)
        cat_f = torch.cat([re_x, re_t], dim=1)
        atten_l = self.gate_rgb(cat_f)
        atten_r = self.gate_t(cat_f)
        atten_v = torch.cat([atten_l, atten_r], dim=1)
        atten_v = self.softmax(atten_v)
        atten1, atten2 = atten_v[:,0:1,:,:], atten_v[:,1:2,:,:]
        merge_out = x * atten1 + t * atten2
        return merge_out
class Teacher(nn.Module):
    def __init__(self, config=cfg):
        super(Teacher, self).__init__()
        self.backbone = p2t_large()
        path = "/home/xug/PycharmProjects/TLD/backbone/P2T/retinanet_p2t_l_fpn_1x_coco-d0ce637b.pth"
        sk = torch.load(path)['state_dict']
        new_state_dice = OrderedDict()
        for k, v in sk.items():
            name = k[9:]
            new_state_dice[name] = v
        self.backbone.load_state_dict(new_state_dice,strict=False)
        self.cfg = config
        self.ms3 = MScoreCA(640)
        self.ms2 = MScoreCA(320)
        self.ms1 = MScoreSA(128)
        self.ms0 = MScoreSA(64)
        self.ssrm0 = SSRM(64, 64)
        self.ssrm1 = SSRM(128, 128)
        self.ssrm2 = SSRM(320, 320)
        self.ssrm3 = SSRM(640, 640)
        self.conv1to3 = BasicConv2d(1,3,1,1,0)
        self.merge1 = merge_out(320,640)
        self.merge2 = merge_out(128,320)
        self.merge3 = merge_out(64,128)
        self.conv64_3 = BasicConv2d(64,1,3,1,1)
        self.up = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.conv512to64 = BasicConv2d(640, 64, 3, 1, 1)
        self.conv256to64 = BasicConv2d(320, 64, 3, 1, 1)
        self.conv128to64 = BasicConv2d(128, 64, 3, 1, 1)
    def forward(self, x, t):
        t = self.conv1to3(t)
        x1_rgb, x2_rgb, x3_rgb, x4_rgb = self.backbone(x)
        x1_t, x2_t, x3_t, x4_t = self.backbone(t)
        ssrm0 = self.ssrm0(x1_rgb, x1_t)
        ssrm1 = self.ssrm1(x2_rgb, x2_t)
        ssrm2 = self.ssrm2(x3_rgb, x3_t)
        ssrm3 = self.ssrm3(x4_rgb, x4_t)
        merge_1 = self.merge1(ssrm2,ssrm3)
        merge_2 = self.merge2(ssrm1,merge_1)
        merge_3 = self.merge3(ssrm0,merge_2)
        out = self.conv64_3(self.up(merge_3))
        ms3 = self.ms3(ssrm3)
        ms2 = self.ms2(ssrm2)
        ms1 = self.ms1(ssrm1)
        ms0 = self.ms0(ssrm0)
        return out, ssrm0, ssrm1, ssrm2, ssrm3, ms3, ms2, ms1, ms0

