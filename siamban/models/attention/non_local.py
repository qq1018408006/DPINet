import math
import torch
import torch.nn as nn
from siamban.core.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D

class SEModule(nn.Module):

    def __init__(self, channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1,
                             padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1,
                             padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        module_input = x
        x = self.avg_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return module_input * x

class nonlocal_channel_similarity(nn.Module):
    def __init__(self, in_channels=256, inter_channels=256):
        super(nonlocal_channel_similarity, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels
        self.group_conv=nn.Sequential(nn.Conv2d(self.in_channels,2*self.in_channels,(self.in_channels,1),1,0,groups=self.in_channels),
                                      nn.ReLU(inplace=True))
        self.dim_adjust=nn.Conv2d(2*self.in_channels,self.in_channels,1,1,0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        batch_size = x.size(0)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        phi_x = phi_x.permute(0, 2, 1)
        f = torch.matmul(theta_x, phi_x).unsqueeze(-1)
        out=self.group_conv(f)
        out=self.dim_adjust(out)
        return out

class NonLocal2D_s_wo_activation(nn.Module):
    def __init__(self,
                 in_channels,
                 reduction=2,
                 use_scale=True,
                 mode='dot_product'):
        super(NonLocal2D_s_wo_activation, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = in_channels // reduction
        self.mode = mode
        self.softmax=nn.Softmax(dim=-1)

        assert mode in ['embedded_gaussian', 'dot_product']
        self.g =nn.Conv2d(self.in_channels, self.inter_channels, 1, 1)
        self.theta = nn.Conv2d(self.in_channels,self.inter_channels,1,1)
        self.phi = nn.Conv2d(self.in_channels,self.inter_channels,1,1)
        self.conv_out=nn.Sequential(
            nn.Conv2d(self.inter_channels, self.in_channels, 1, 1,bias=False),
            nn.BatchNorm2d(self.in_channels)
        )

    def dot_product(self, theta_x, phi_x):
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight=self.softmax(pairwise_weight)
        return pairwise_weight

    def forward(self, x):
        n, c, h, w = x.shape
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(n, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(n, self.inter_channels, -1)

        pairwise_func = getattr(self, self.mode)
        pairwise_weight = pairwise_func(theta_x, phi_x)/math.sqrt(c)
        y = torch.matmul(pairwise_weight, g_x)
        y = y.permute(0, 2, 1).reshape(n, self.inter_channels, h, w)
        y=self.conv_out(y)
        return y+x

class NonLocal2D_s_wo_activation_and_res(nn.Module):
    def __init__(self,
                 in_channels,
                 reduction=2,
                 use_scale=True,
                 mode='dot_product'):
        super(NonLocal2D_s_wo_activation_and_res, self).__init__()
        self.in_channels = in_channels
        self.reduction = reduction
        self.use_scale = use_scale
        self.inter_channels = in_channels // reduction
        self.mode = mode
        self.softmax=nn.Softmax(dim=-1)

        assert mode in ['embedded_gaussian', 'dot_product']
        self.g =nn.Conv2d(self.in_channels, self.inter_channels, 1, 1)
        self.theta = nn.Conv2d(self.in_channels,self.inter_channels,1,1)
        self.phi = nn.Conv2d(self.in_channels,self.inter_channels,1,1)
        self.conv_out=nn.Sequential(
            nn.Conv2d(self.inter_channels, self.in_channels, 1, 1,bias=False),
            nn.BatchNorm2d(self.in_channels)
        )

    def dot_product(self, theta_x, phi_x):
        pairwise_weight = torch.matmul(theta_x, phi_x)
        pairwise_weight=self.softmax(pairwise_weight)
        return pairwise_weight

    def forward(self, x):
        n, c, h, w = x.shape
        g_x = self.g(x).view(n, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(n, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(n, self.inter_channels, -1)
        pairwise_func = getattr(self, self.mode)
        pairwise_weight = pairwise_func(theta_x, phi_x)/math.sqrt(c)
        y = torch.matmul(pairwise_weight, g_x)
        y = y.permute(0, 2, 1).reshape(n, self.inter_channels, h, w)
        y=self.conv_out(y)
        return y

class AttnAllLayer(nn.Module):
    def __init__(self,in_channels,type):
        super(AttnAllLayer, self).__init__()
        self.num=len(in_channels)
        if type == 'cse':
            for i in range(self.num):
                self.add_module('downsample' + str(i + 2),
                                cse(in_channels[i]))
        else:
            raise ValueError('wrong attn type!')

    def forward(self,z_feats, x_feats, mask=None,box=None):
        z_out = []
        x_out = []
        for i in range(self.num):
            adj_layer = getattr(self, 'downsample'+str(i+2))
            out=adj_layer(z_feats[i],x_feats[i],mask,box=box)

            z_out.append(out[0])
            x_out.append(out[1])
        return z_out, x_out

    def init(self,zf, mask,box):
        z_out = []
        for i in range(self.num):
            adj_layer = getattr(self, 'downsample' + str(i + 2))
            if mask is not None:
                out = adj_layer.init(zf[i], mask,box=box)
            else:
                out = adj_layer.init(zf[i],box=box)
            z_out.append(out)
        return z_out

    def track(self, xf):
        x_out = []
        for i in range(self.num):
            adj_layer = getattr(self, 'downsample' + str(i + 2))
            out = adj_layer.track(xf[i])
            x_out.append(out)
        return x_out

    def track2(self, xf):
        x_out = []
        z_out =[]
        for i in range(self.num):
            adj_layer = getattr(self, 'downsample' + str(i + 2))
            try:
                out1, out2=adj_layer.track(xf[i])
                z_out.append(out1)
                x_out.append(out2)
                return z_out ,x_out
            except:
                out = adj_layer.track(xf[i])
                x_out.append(out)
                return x_out

class cse(nn.Module):
    def __init__(self, in_channels=256, out_channels=256):
        super(cse, self).__init__()
        self.non_local = NonLocal2D_s_wo_activation_and_res(in_channels, reduction=2)
        # channel cues
        self.prpool = PrRoIPool2D(7, 7,15 / 127)
        self.chan_z = nonlocal_channel_similarity()
        self.maxpool_z = nn.Sequential(nn.AdaptiveMaxPool2d(1),
                                     nn.Conv2d(in_channels, in_channels // 2, kernel_size=1,padding=0),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_channels // 2, in_channels, kernel_size=1,padding=0),)
        self.avgpool_z = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, padding=0),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(in_channels // 2, in_channels, kernel_size=1, padding=0), )
        self.dim_reduction1 = nn.Conv2d( in_channels, in_channels,(3,1),1,0,groups=in_channels)

        self.sigmoid = nn.Sigmoid()
        # spatial cues
        mask_size = 15
        self.adjust_conv = nn.Sequential(
            nn.Conv2d(2, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Upsample(size=mask_size - 1, mode='bilinear', align_corners=True),
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.Upsample(size=mask_size, mode='bilinear', align_corners=True),
            nn.Conv2d(4, 1, 3, 1, 1),
        )
        self.phi = nn.Conv2d(in_channels, in_channels, 1, 1)
        self.delta = nn.Conv2d(in_channels, in_channels, 1, 1)

    def forward(self, zf, xf, mask=None, box=None):
        mask_bg = 1 - mask
        mask = torch.cat((mask, mask_bg), dim=1)
        mask = self.adjust_conv(mask)

        zf_raw = zf[1]
        shape_z_raw = zf_raw.shape
        shape_x = xf.shape
        # self attn
        self_attn = self.non_local(zf_raw)
        # cahnnel cues
        zf_raw_detach=zf_raw.detach()
        b=zf_raw_detach.shape[0]
        _bboxes = torch.zeros(b, 5)
        _bboxes[:, 0] = torch.tensor(range(0, b))
        _bboxes[:, 1:] = box
        rois=self.prpool(zf_raw_detach, _bboxes.cuda())

        z_avg = self.avgpool_z(rois)
        z_max = self.maxpool_z(rois)
        z_2n_order = self.chan_z(rois)

        channels_cues_z = torch.cat((z_avg, z_max, z_2n_order), dim=2)
        channels_cues_z = self.sigmoid(self.dim_reduction1(channels_cues_z))

        # spatial cues
        z_phi = self.phi(zf_raw)
        x_delta = self.delta(xf)
        z_phi = z_phi.view(-1, shape_z_raw[1], shape_z_raw[2] * shape_z_raw[3])
        x_delta = x_delta.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1)
        out_spatial_cues = torch.matmul(x_delta, z_phi) / math.sqrt(shape_x[1])
        mask = mask.view(-1, 1, shape_z_raw[2] * shape_z_raw[3]).permute(0, 2, 1)
        out_spatial_cues = torch.matmul(out_spatial_cues, mask).permute(0, 2, 1).view(-1, 1, shape_x[2],
                                                                                      shape_x[3])
        out_spatial_cues = torch.clamp(out_spatial_cues, 0, 1)

        zf_out = channels_cues_z * zf_raw + zf_raw + self_attn
        xf_out = channels_cues_z * xf + xf - out_spatial_cues

        zf_raw = zf_out
        if zf_out.size(3) < 20:
            l = 4
            r = l + 7
            zf_out = zf_out[:, :, l:r, l:r]
        return (zf_out, zf_raw), xf_out

    def init(self, zf, mask=None,box=None):
        mask_bg = 1 - mask
        mask = torch.cat((mask, mask_bg), dim=0).unsqueeze(0)
        mask = self.adjust_conv(mask)

        zf_raw = zf[1]
        shape_z_raw = zf_raw.shape
        # self attn
        self_attn = self.non_local(zf_raw)
        # cahnnel cues
        zf_raw_detach = zf_raw.detach()
        b = zf_raw_detach.shape[0]
        _bboxes = torch.zeros(b, 5)
        _bboxes[:, 0] = torch.tensor(range(0, b))
        _bboxes[:, 1:] = box
        rois = self.prpool(zf_raw_detach, _bboxes.cuda())

        z_avg = self.avgpool_z(rois)
        z_max = self.maxpool_z(rois)
        z_2n_order = self.chan_z(rois)
        channels_cues_z = torch.cat((z_avg, z_max, z_2n_order), dim=2)
        channels_cues_z = self.dim_reduction1(channels_cues_z)
        channels_cues_z = self.sigmoid(channels_cues_z)
        self.channels_cues_z = channels_cues_z

        # spatial cues
        z_phi = self.phi(zf_raw)
        self.z_phi = z_phi.view(-1, shape_z_raw[1], shape_z_raw[2] * shape_z_raw[3])
        self.mask = mask.view(-1, 1, shape_z_raw[2] * shape_z_raw[3]).permute(0, 2, 1)

        zf_out = zf_raw * channels_cues_z + zf_raw + self_attn
        zf_raw = zf_out
        if zf_out.size(3) < 20:
            l = 4
            r = l + 7
            zf_out = zf_out[:, :, l:r, l:r]

        return zf_out, zf_raw

    def track(self, xf):
        # spatial cues
        shape_x = xf.shape
        x_delta = self.delta(xf)
        x_delta = x_delta.view(-1, shape_x[1], shape_x[2] * shape_x[3]).permute(0, 2, 1)
        out_spatial_cues = torch.matmul(x_delta, self.z_phi) / math.sqrt(shape_x[1])
        out_spatial_cues = torch.matmul(out_spatial_cues, self.mask).permute(0, 2, 1).view(-1, 1, shape_x[2],
                                                                                           shape_x[3])
        out_spatial_cues = torch.clamp(out_spatial_cues, 0, 1)
        xf_out = self.channels_cues_z * xf + xf - out_spatial_cues
        return xf_out

class MDI(nn.Module):
    def __init__(self, pool_size=7, use_post_corr=True):
        super(MDI,self).__init__()
        num_corr_channel = pool_size*pool_size
        self.use_post_corr = use_post_corr
        if use_post_corr:
            self.post_corr = nn.Sequential(
                nn.Conv2d(num_corr_channel, 128, kernel_size=(1, 1), padding=0, stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 128, kernel_size=(1, 1), padding=0, stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.Conv2d(128, 256, kernel_size=(1, 1), padding=0, stride=1),
                nn.BatchNorm2d(256),
            )
        self.channel_attention = SEModule(256,reduction=4)
        self.selfattn=NonLocal2D_s_wo_activation(in_channels=256)

    def fuse_feat(self, feat2,feat1):
        feat_corr, _ = self.corr_fun(feat1, feat2)
        if self.use_post_corr:
            feat_corr = self.post_corr(feat_corr)
        feat_ca = self.channel_attention(feat_corr)
        feat_ca = self.selfattn(feat_ca)
        return feat_ca

    def corr_fun(self, ker, feat):
        return self.corr_fun_mat(ker, feat)

    def corr_fun_mat(self, ker, feat):
        b, c, h, w = feat.shape
        ker = ker.reshape(b, c, -1).transpose(1, 2)
        feat = feat.reshape(b, c, -1)
        corr = torch.matmul(ker, feat)
        corr = corr.reshape(*corr.shape[:2], h, w)
        return corr, ker

    def forward(self,zf,xf):
        return self.fuse_feat(zf,xf)
