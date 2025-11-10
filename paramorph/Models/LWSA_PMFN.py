'''
LWSA module

A partial code was retrieved from:
https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration

Swin-Transformer code was retrieved from:
https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation

Original Swin-Transformer paper:
Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021).
Swin transformer: Hierarchical vision transformer using shifted windows.
arXiv preprint arXiv:2103.14030.
'''

import Models.basic_LWSA as basic
import Models.Conv3dReLU as Conv3dReLU
import torch.nn as nn
import utils.configs_TransMatch as configs
import torch

class LWSA(nn.Module):
    def __init__(self, config):
        super(LWSA, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = basic.SwinTransformer(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf
                                           )
        self.c1 = Conv3dReLU.Conv3dReLU(1, embed_dim//2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU.Conv3dReLU(1, config.reg_head_chan, 3, 1, use_batchnorm=False)

        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

        self.conv1 = conv_block(1)
        self.down1 = down_block(1, embed_dim//2)
        self.conv2 = conv_block(embed_dim//2)
        self.down2 = down_block(embed_dim//2, embed_dim)
        self.conv3 = conv_block(embed_dim)
        self.down3 = down_block(embed_dim, embed_dim*2)
        self.conv4 = conv_block(embed_dim*2)
        self.down4 = down_block(embed_dim*2, embed_dim*4)
        self.conv5 = conv_block(embed_dim*4)
        self.down5 = down_block(embed_dim*4, embed_dim*8)
        self.Conv_1x1 = nn.Conv3d(1, config.reg_head_chan, kernel_size=1, stride=1, padding=0)

        self.fusion0 = nn.Conv3d(embed_dim*16, embed_dim*8, kernel_size=1, stride=1, padding=0)
        self.fusion1 = nn.Conv3d(embed_dim * 8, embed_dim * 4, kernel_size=1, stride=1, padding=0)
        self.fusion2 = nn.Conv3d(embed_dim * 4, embed_dim * 2, kernel_size=1, stride=1, padding=0)
        self.fusion3 = nn.Conv3d(embed_dim * 2, embed_dim, kernel_size=1, stride=1, padding=0)



    def forward(self, x):
        # print('The shape of x(Input of transformer function):', x.shape)
        source = x[:, 0:1, :, :, :]
        #print('The shape of source:', source.shape)
        if self.if_convskip:
            x_s0 = x.clone()
            cov0 = self.conv1(x_s0)
            cov1 = self.down1(cov0)
            #cov0 = self.Conv_1x1(cov0)

            cov2 = self.conv2(cov1)
            cov2 = self.down2(cov2)

            cov3 = self.conv3(cov2)
            cov3 = self.down3(cov3)

            cov4 = self.conv4(cov3)
            cov4 = self.down4(cov4)

            cov5 = self.conv5(cov4)
            cov5 = self.down5(cov5)
            # x_s1 = self.avg_pool(x)
            # f4 = cov1()
            # f5 = cov0()
        else:
            f4 = None
            f5 = None

        out_feats = self.transformer(x)

        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
            f3 = out_feats[-4]

        else:
            f1 = None
            f2 = None
            f3 = None
        '''
        fu0 = torch.cat([out_feats[-1], cov5], dim=1)
        fu0 = self.fusion0(fu0)
        fu1 = torch.cat([f1, cov4], dim=1)
        fu1 = self.fusion1(fu1)
        fu2 = torch.cat([f2, cov3], dim=1)
        fu2 = self.fusion2(fu2)
        fu3 = torch.cat([f3, cov2], dim=1)
        fu3 = self.fusion3(fu3)
        '''

        fu0 = torch.add(out_feats[-1], cov5)
        fu1 = torch.add(f1, cov4)
        fu2 = torch.add(f2, cov3)
        fu3 = torch.add(f3, cov2)
        return fu3, fu2, fu1, fu0


CONFIGS = {
    'TransMatch_LPBA40': configs.get_TransMatch_LPBA40_config()
}



class down_block(nn.Module):
    def __init__(self, c_in,out_channels, scale=16, k_size=3):
        super().__init__()
        self.dw_conv = nn.Conv3d(c_in, c_in, kernel_size=k_size, groups=c_in, stride=2, padding=(k_size - 1) // 2,
                                 bias=False)
        self.norm = nn.GroupNorm(c_in, c_in)
        self.expansion = nn.Conv3d(c_in, scale * c_in, kernel_size=(1, 1, 1), stride=1)
        self.act = nn.GELU()
        self.compress = nn.Conv3d(scale * c_in, out_channels, kernel_size=(1, 1, 1), stride=1)
        self.shortcut = nn.Conv3d(c_in, out_channels, kernel_size=(1, 1, 1), stride=2)

    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.norm(self.dw_conv(x))
        out = self.act(self.expansion(out))
        out = self.compress(out)
        out = torch.add(out, shortcut)
        return out


#mednext 的基本conv
class conv_block(nn.Module):
    def __init__(self, c_in, scale=16, k_size=3):
        super().__init__()
        self.dw_conv = nn.Conv3d(c_in, c_in, kernel_size=k_size, groups=c_in, stride=1,
                                 padding=(k_size - 1) // 2, bias=False)
        self.expansion = nn.Conv3d(c_in, c_in * scale, kernel_size=(1, 1, 1), stride=1)
        self.act = nn.GELU()
        self.compress = nn.Conv3d(c_in * scale, c_in, kernel_size=(1, 1, 1), stride=1)
        self.norm = nn.GroupNorm(c_in, c_in)

    def forward(self, x):
        identity = x
        out = self.norm(self.dw_conv(x))
        out = self.act(self.expansion(out))
        out = self.compress(out)
        out = torch.add(out, identity)
        return out



#Mednext的decoder
class DecoderBlock2(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()

        self.dw_conv = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, stride=2, padding=(3 - 1) // 2,
                                          output_padding=1, groups=in_channels, bias=False)
        self.norm = nn.GroupNorm(in_channels, in_channels)
        self.expansion = nn.Conv3d(in_channels, in_channels * 16, kernel_size=(1, 1, 1), stride=1)
        self.act = nn.GELU()
        self.compress = nn.Conv3d(in_channels * 16, in_channels//2, kernel_size=(1, 1, 1), stride=1)
        self.shortcut = nn.ConvTranspose3d(in_channels, in_channels//2, kernel_size=(1, 1, 1), output_padding=1, stride=2)


        self.conv1 = Conv3dReLU(
            in_channels//2 + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )

        self.conv = conv_block(in_channels)

    def forward(self, x, skip=None):
        x = self.conv(x)
        short = self.shortcut(x)
        x = self.norm(self.dw_conv(x))
        x = self.act(self.expansion(x))
        x = self.compress(x)
        x = torch.add(x, short)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DecoderBlock3(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()

        self.dw_conv = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=3, stride=2, padding=(3 - 1) // 2,
                                          output_padding=1, groups=in_channels, bias=False)
        self.norm = nn.GroupNorm(in_channels, in_channels)
        self.expansion = nn.Conv3d(in_channels, in_channels * 16, kernel_size=(1, 1, 1), stride=1)
        self.act = nn.GELU()
        self.compress = nn.Conv3d(in_channels * 16, out_channels, kernel_size=(1, 1, 1), stride=1)
        self.shortcut = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(1, 1, 1), output_padding=1, stride=2)

        self.conv = conv_block(out_channels)


    def forward(self, x, skip=None):

        short = self.shortcut(x)
        x = self.norm(self.dw_conv(x))
        x = self.act(self.expansion(x))
        x = self.compress(x)
        x = torch.add(x, short)
        if skip is not None:
            x = torch.add(x, skip)
        #x = self.conv(x)
        return x




