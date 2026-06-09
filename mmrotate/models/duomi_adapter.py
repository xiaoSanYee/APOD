import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward, DWTInverse

class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size,
                                   stride, padding, groups=in_channels, bias=bias)

        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
class BB(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=3 // 2, groups=dim)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=3 // 2, groups=dim)

    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        out = x + res
        return out
class SobelOperator(nn.Module):

    def __init__(self, in_channels=3):
        super().__init__()
        self.sobel_x = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False
        )

        self.sobel_y = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channels,
            bias=False
        )

        self._init_weights()

        for param in self.parameters():
            param.requires_grad = False

    def _init_weights(self):
        sobel_x_weight = torch.tensor([
            [1.0, 2.0, 1.0],
            [0.0, 0.0, 0.0],
            [-1.0, -2.0, -1.0]
        ])
        sobel_y_weight = torch.tensor([
            [1.0, 0.0, -1.0],
            [2.0, 0.0, -2.0],
            [1.0, 0.0, -1.0]
        ])

        sobel_x_weight = sobel_x_weight.repeat(self.sobel_x.in_channels, 1, 1, 1)
        sobel_y_weight = sobel_y_weight.repeat(self.sobel_y.in_channels, 1, 1, 1)

        self.sobel_x.weight.data = sobel_x_weight
        self.sobel_y.weight.data = sobel_y_weight

    def forward(self, x):

        edge_x = self.sobel_x(x)
        edge_y = self.sobel_y(x)
        return edge_x, edge_y


class EdgeEnhancer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.down = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0),
            nn.SiLU()
        )

    def forward(self, x):
        x = x + self.down(x)
        x = self.conv(x)
        out = x
        return out

class EdgeEnhancementModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.sobel = SobelOperator(in_channels=dim)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0),
            nn.SiLU()
        )
        self.ee = EdgeEnhancer(dim)

    def forward(self, x):
        res = x
        x1, x2 = self.sobel(x)
        edge_strength = torch.sqrt(x1 ** 2 + x2 ** 2 + 1e-8)
        x = self.conv(edge_strength)
        x = self.ee(x)
        out = x + res
        return out


class DetailRestorationModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0),
            nn.SiLU()
        )
        hid_dim = dim // 2
        self.hid_dim = hid_dim
        self.down1 = nn.Conv2d(hid_dim, hid_dim, kernel_size=3, stride=2, padding=1, groups=hid_dim)
        self.down2 = nn.Conv2d(hid_dim, hid_dim, kernel_size=3, stride=4, padding=1, groups=hid_dim)

        self.conv_l = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv_h = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.norm_l = nn.BatchNorm2d(dim)
        self.norm_h = nn.BatchNorm2d(dim)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1, stride=1, padding=0),
            nn.SiLU()
        )

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.conv(x)
        x1, x2 = torch.split(x, [self.hid_dim, self.hid_dim], dim=1)
        dx1 = self.down1(x1)
        dx2 = self.down2(x2)
        udx1 = F.interpolate(dx1, size=(H, W), mode='bilinear', align_corners=False)
        udx2 = F.interpolate(dx2, size=(H, W), mode='bilinear', align_corners=False)
        udx = torch.cat([udx1, udx2], dim=1)
        lx = self.norm_l(self.conv_l(self.act(x * udx)))
        hx = self.norm_h(self.conv_h(self.act(x - udx)))
        out = self.proj(torch.cat([lx, hx], dim=1))
        return out


class FeatureAlignmentRestorationOperator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = nn.BatchNorm2d(dim)
        self.dw_conv1 = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=3 // 2, stride=1, groups=dim),
            nn.SiLU()
        )
        self.frm = DetailRestorationModule(dim=dim)
        self.eem = EdgeEnhancementModule(dim=dim)
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=1, padding=0, stride=1),
            nn.SiLU()
        )
        self.mlp = BB(dim=dim)

    def forward(self, x):
        res = x
        x = x + self.norm(self.dw_conv1(x))
        f = self.frm(x)
        e = self.eem(x)
        x = torch.cat([f, e], dim=1)
        x = self.proj(x)
        y = x + res

        out = self.mlp(y)

        return out


class WaveletMLP(nn.Module):
    def __init__(self, in_channel, expand=1, wave='haar', mode='zero'):
        super(WaveletMLP, self).__init__()

        self.dwt = DWTForward(J=1, wave=wave, mode=mode)
        self.idwt = DWTInverse(wave=wave, mode=mode)

        self.process_ll = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=int(expand * in_channel), kernel_size=1, stride=1,
                      padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels=int(expand * in_channel), out_channels=in_channel, kernel_size=1, stride=1,
                      padding=0))

        self.process_hf = nn.Sequential(
            nn.Conv2d(in_channel, int(expand * in_channel), 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(int(expand * in_channel), in_channel, 1)
        )

        self.hf_weights = nn.Parameter(torch.ones(3, 1, 1, 1, 1))

    def forward(self, x):
        B, C, H, W = x.shape

        yl, yh = self.dwt(x)

        ll_processed = self.process_ll(yl)

        h_lh = yh[0][:, :, 0, :, :]
        h_hl = yh[0][:, :, 1, :, :]
        h_hh = yh[0][:, :, 2, :, :]

        h_lh_processed = self.process_hf(h_lh) * self.hf_weights[0]
        h_hl_processed = self.process_hf(h_hl) * self.hf_weights[1]
        h_hh_processed = self.process_hf(h_hh) * self.hf_weights[2]

        hf_processed = torch.stack([h_lh_processed, h_hl_processed, h_hh_processed], dim=2)
        yh_processed = [hf_processed]

        x_out = self.idwt((ll_processed, yh_processed))
        x_out = x_out[:, :, :H, :W]
        return x_out


class MultiScaleWaveletFusionOperator(nn.Module):
    def __init__(self, dim):
        super(MultiScaleWaveletFusionOperator, self).__init__()
        self.dilated_conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, dilation=1, groups=dim)
        self.wave_mlp = WaveletMLP(in_channel=dim)

    def forward(self, x):
        res1 = x
        d1 = self.dilated_conv1(x)
        fusion = d1 + res1
        res2 = fusion
        w = self.wave_mlp(fusion)
        out = w + res2

        return out


class Adapter_single(nn.Module):
    def __init__(self, dim, adapter_dim=64, sort=None):
        super().__init__()
        self.sort = sort
        self.down_project = nn.Linear(dim, adapter_dim)
        self.act = F.gelu
        self.dropout = nn.Dropout(p=0.1)
        self.up_project = nn.Linear(adapter_dim, dim)

        self.norm = nn.LayerNorm(dim)
        self.gamma = nn.Parameter(torch.ones(dim) * 1e-6)
        self.gammax = nn.Parameter(torch.ones(dim))

        if sort == 'FARA':
            self.operator = FeatureAlignmentRestorationOperator(adapter_dim)
        elif sort == 'MSWFA':
            self.operator = MultiScaleWaveletFusionOperator(adapter_dim)

    def forward(self, x, h, w):
        res = x

        if self.norm_type == 'scale':
            x = self.scale_input * self.norm(x) + self.shift_input
        else:
            x = self.norm(x) * self.gamma + x * self.gammax

        x_down = self.down_project(x)
        b, n, c = x_down.shape
        x_down = x_down.reshape(b, h, w, c).permute(0, 3, 1, 2)

        x_down = self.operator(x_down)

        x_down = x_down.permute(0, 2, 3, 1).reshape(b, n, c)
        x_nonlinear = self.act(x_down)
        x_nonlinear = self.dropout(x_nonlinear)
        x_up = self.up_project(x_nonlinear)

        return res + x_up


class SwinSimAM(nn.Module):
    def __init__(self, epsilon=1e-8, local_window=3):
        super(SwinSimAM, self).__init__()
        self.epsilon = epsilon
        self.local_window = local_window

        assert local_window % 2 == 1, "local_window must be odd"
        self.pad = (local_window - 1) // 2

    def forward(self, x):

        B, C, H, W = x.shape


        if C > 1:
            x = x.mean(dim=1, keepdim=True)

        unfold = nn.Unfold(kernel_size=self.local_window, padding=self.pad, stride=1)
        x_unfold = unfold(x)
        x_unfold = x_unfold.transpose(1, 2)

        local_mu = x_unfold.mean(dim=2, keepdim=True)
        local_sigma = x_unfold.var(dim=2, keepdim=True) + self.epsilon

        local_mu = local_mu.transpose(1, 2).view(B, 1, H, W)
        local_sigma = local_sigma.transpose(1, 2).view(B, 1, H, W)

        energy = torch.exp(-(x - local_mu) ** 2 / (2 * local_sigma))

        attention = energy / (energy.sum(dim=(2, 3), keepdim=True) + self.epsilon)

        return attention


class CA(nn.Module):
    def __init__(self, dim, reduction=4):
        super(CA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1, bias=False),
            nn.SiLU(),
            nn.Conv2d(dim // reduction, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(self.avg_pool(x))


class SemanticGuidedFeatureRefinementOperator(nn.Module):
    def __init__(self, dim, reduction=4):
        super(SemanticGuidedFeatureRefinementOperator, self).__init__()

        self.y_align = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

        self.cpcs = nn.Sequential(
            DepthwiseSeparableConv(in_channels=2 * dim, out_channels=dim, kernel_size=3, padding=1, stride=1),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.SiLU()
        )

        self.chan_attn = CA(dim, reduction)
        self.spat_attn = SwinSimAM(local_window=3)

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.SiLU()
        )
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, y):

        input_size = x.shape[2:]

        y_up = F.interpolate(y, size=input_size, mode='bilinear', align_corners=False)
        y_up = self.y_align(y_up)

        aggregation = self.cpcs(torch.cat([x, y_up], dim=1))

        c_attn = self.chan_attn(aggregation)
        aggregation = aggregation * c_attn
        s_attn = self.spat_attn(aggregation)

        fused = (1 - s_attn) * y_up + s_attn * x

        fused = x + self.gamma * self.fuse_conv(fused)

        return fused

class Adapter_dual(nn.Module):
    def __init__(self, x_dim, y_dim, adapter_dim=64, sort='SFRA'):
        super().__init__()

        self.down_project1 = nn.Linear(x_dim, adapter_dim)
        self.down_project2 = nn.Linear(y_dim, adapter_dim)

        self.act = F.gelu
        self.dropout = nn.Dropout(p=0.1)
        self.up_project = nn.Linear(adapter_dim, x_dim)

        self.norm_1 = nn.LayerNorm(x_dim)
        self.gamma_1 = nn.Parameter(torch.ones(x_dim) * 1e-6)
        self.gammax_1 = nn.Parameter(torch.ones(x_dim))
        self.norm_2 = nn.LayerNorm(y_dim)
        self.gamma_2 = nn.Parameter(torch.ones(y_dim) * 1e-6)
        self.gammax_2 = nn.Parameter(torch.ones(y_dim))

        if sort == "SFRO4":
            self.operator = SemanticGuidedFeatureRefinementOperator(dim=adapter_dim)


    def forward(self, x, y, h, w):
        res = x
        x = self.norm_1(x) * self.gamma_1 + x * self.gammax_1
        y = self.norm_2(y) * self.gamma_2 + y * self.gammax_2

        x_down = self.down_project1(x)
        y_down = self.down_project2(y)

        b, n, c = x_down.shape

        x_down = x_down.reshape(b, h, w, c).permute(0, 3, 1, 2)
        y_down = y_down.reshape(b, h, w, c).permute(0, 3, 1, 2)

        x_down = self.operator(x_down, y_down)

        x_down = x_down.permute(0, 2, 3, 1).reshape(b, n, c)

        x_nonlinear = self.act(x_down)
        x_nonlinear = self.dropout(x_nonlinear)
        x_up = self.up_project(x_nonlinear)

        return res + x_up
