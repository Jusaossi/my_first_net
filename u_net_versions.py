import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Encoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder_block, self).__init__()
        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1)
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        x = self.convr1(x)
        x = self.convr2(x)
        skip = x  # save x to use in concat in the Decoder path
        out = self.maxPool(x)
        return out, skip


class Decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_size, padding, drop_out=False):
        super(Decoder_block, self).__init__()

        self.drop_out = drop_out
        self.dropout_layer = nn.Dropout2d(p=0.5)

        self.convr1 = ConvBnRelu(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding)
        self.convr2 = ConvBnRelu(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding)
        self.upSample = nn.ConvTranspose2d(in_channels, out_channels, upsample_size, stride=2)

    def _crop_concat(self, upsampled, downsampled):
        """
         pad upsampled to the (h, w) of downsampled to concatenate
         for expansive path.
        Returns:
            The concatenated tensor
        """
        h = downsampled.size()[2] - upsampled.size()[2]
        w = downsampled.size()[3] - upsampled.size()[3]
        upsampled = F.pad(upsampled, (0, w, 0, h))
        return torch.cat((downsampled, upsampled), 1)

    def forward(self, x, down_tensor):
        x = self.upSample(x)
        x = self._crop_concat(x, down_tensor)
        x = self.convr1(x)
        if self.drop_out:
            x = self.dropout_layer(x)
        x = self.convr2(x)
        if self.drop_out:
            x = self.dropout_layer(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.down1 = Encoder_block(1, 64)
        self.down2 = Encoder_block(64, 128)
        self.down3 = Encoder_block(128, 256)
        self.down4 = Encoder_block(256, 512)

        self.center = nn.Sequential(
            ConvBnRelu(512, 1024, kernel_size=(3, 3), stride=1, padding=1),
            ConvBnRelu(1024, 1024, kernel_size=(3, 3), stride=1, padding=1)
        )

        self.up1 = Decoder_block(in_channels=1024, out_channels=512, upsample_size=2, padding=1)
        self.up2 = Decoder_block(in_channels=512, out_channels=256, upsample_size=2, padding=1)
        self.up3 = Decoder_block(in_channels=256, out_channels=128, upsample_size=2, padding=1)
        self.up4 = Decoder_block(in_channels=128, out_channels=64, upsample_size=2, padding=1)

        # 1x1 convolution at the last layer
        self.outputNN = nn.Conv2d(64, 1, kernel_size=(1, 1), padding=0, stride=1)

        # self._initialize_weights()

    def forward(self, x):
        # x.cuda(device)
        x, skip1 = self.down1(x)
        # print(x.shape)
        x, skip2 = self.down2(x)
        # print(x.shape)
        x, skip3 = self.down3(x)
        # print(x.shape)
        x, skip4 = self.down4(x)
        # print(x.shape)
        x = self.center(x)
        # print(x.shape)
        x = self.up1(x, skip4)
        # print(x.shape)
        x = self.up2(x, skip3)
        # print(x.shape)
        x = self.up3(x, skip2)
        # print(x.shape)
        x = self.up4(x, skip1)
        # print(x.shape)
        x = self.outputNN(x)
        x = torch.sigmoid(x)
        return x


class UNetHalf(nn.Module):
    def __init__(self):
        super(UNetHalf, self).__init__()
        self.down1 = Encoder_block(1, 32)
        self.down2 = Encoder_block(32, 64)
        self.down3 = Encoder_block(64, 128)
        self.down4 = Encoder_block(128, 256)

        self.center = nn.Sequential(
            ConvBnRelu(256, 512, kernel_size=(3, 3), stride=1, padding=1),
            ConvBnRelu(512, 512, kernel_size=(3, 3), stride=1, padding=1)
        )

        self.up1 = Decoder_block(in_channels=512, out_channels=256, upsample_size=2, padding=1)
        self.up2 = Decoder_block(in_channels=256, out_channels=128, upsample_size=2, padding=1)
        self.up3 = Decoder_block(in_channels=128, out_channels=64, upsample_size=2, padding=1)
        self.up4 = Decoder_block(in_channels=64, out_channels=32, upsample_size=2, padding=1)

        # 1x1 convolution at the last layer
        self.outputNN = nn.Conv2d(32, 1, kernel_size=(1, 1), padding=0, stride=1)

        # self._initialize_weights()

    def forward(self, x):
        # x.cuda(device)
        x, skip1 = self.down1(x)
        # print(x.shape)
        x, skip2 = self.down2(x)
        # print(x.shape)
        x, skip3 = self.down3(x)
        # print(x.shape)
        x, skip4 = self.down4(x)
        # print(x.shape)
        x = self.center(x)
        # print(x.shape)
        x = self.up1(x, skip4)
        # print(x.shape)
        x = self.up2(x, skip3)
        # print(x.shape)
        x = self.up3(x, skip2)
        # print(x.shape)
        x = self.up4(x, skip1)
        # print(x.shape)
        x = self.outputNN(x)
        x = torch.sigmoid(x)
        return x


class UNetQuarter(nn.Module):
    def __init__(self):
        super(UNetQuarter, self).__init__()
        self.down1 = Encoder_block(1, 16)
        self.down2 = Encoder_block(16, 32)
        self.down3 = Encoder_block(32, 64)
        self.down4 = Encoder_block(64, 128)

        self.center = nn.Sequential(
            ConvBnRelu(128, 256, kernel_size=(3, 3), stride=1, padding=1),
            ConvBnRelu(256, 256, kernel_size=(3, 3), stride=1, padding=1)
        )

        self.up1 = Decoder_block(in_channels=256, out_channels=128, upsample_size=2, padding=1)
        self.up2 = Decoder_block(in_channels=128, out_channels=64, upsample_size=2, padding=1)
        self.up3 = Decoder_block(in_channels=64, out_channels=32, upsample_size=2, padding=1)
        self.up4 = Decoder_block(in_channels=32, out_channels=16, upsample_size=2, padding=1)

        # 1x1 convolution at the last layer
        self.outputNN = nn.Conv2d(16, 1, kernel_size=(1, 1), padding=0, stride=1)

        # self._initialize_weights()

    def forward(self, x):
        # x.cuda(device)
        x, skip1 = self.down1(x)
        # print(x.shape)
        x, skip2 = self.down2(x)
        # print(x.shape)
        x, skip3 = self.down3(x)
        # print(x.shape)
        x, skip4 = self.down4(x)
        # print(x.shape)
        x = self.center(x)
        # print(x.shape)
        x = self.up1(x, skip4)
        # print(x.shape)
        x = self.up2(x, skip3)
        # print(x.shape)
        x = self.up3(x, skip2)
        # print(x.shape)
        x = self.up4(x, skip1)
        # print(x.shape)
        x = self.outputNN(x)
        x = torch.sigmoid(x)
        return x

class Gated_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, b_norm, affine, dropout):
        super(Gated_block, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        #self.convy = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
        #self.bn = nn.BatchNorm2d(out_channels)
        self.ins = nn.InstanceNorm2d(out_channels, affine=affine)
        self.relu = nn.ReLU(inplace=True)
        self.sig = nn.Sigmoid()
        if dropout is not None:
            self.dropout = nn.Dropout2d(inplace=True)
        else:
            self.dropout = None

        #self.b_norm = b_norm

    def forward(self, x):
        x1 = self.conv(x)
        if self.dropout is not None:
            x1 = self.dropout(x1)
        x1 = self.relu(x1)
        x2 = self.conv(x)
        if self.dropout is not None:
            x2 = self.dropout(x2)
        x2 = self.sig(x2)
        x = x1*x2
        x = self.ins(x)
        return x

class Encoder_block_gated(nn.Module):
    def __init__(self, in_channels, out_channels, b_norm, affine, dropout):
        super(Encoder_block_gated, self).__init__()
        self.convr1 = Gated_block(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1, b_norm=b_norm, affine=affine, dropout=dropout)
        self.convr2 = Gated_block(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=1, b_norm=b_norm, affine=affine, dropout=dropout)
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

    def forward(self, x):
        x = self.convr1(x)
        x = self.convr2(x)
        skip = x             # save x to use in concat in the Decoder path
        x = self.maxPool(x)
        return x, skip


class Decoder_block_gated(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_factor, padding, b_norm, affine, bilinear, dropout):
        super(Decoder_block_gated, self).__init__()
        self.bilinear = bilinear
        if self.bilinear:
            self.upSample = nn.Upsample(scale_factor=upsample_factor, mode='bilinear', align_corners=True)
        else:
            self.upSample = nn.ConvTranspose2d(in_channels, out_channels, upsample_factor, stride=2)
        self.conv = Gated_block(in_channels, in_channels//2, kernel_size=(3, 3), stride=1, padding=padding, b_norm=b_norm,
                                affine=affine, dropout=dropout)  # extra conv, will remove later
        self.convr1 = Gated_block(in_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding, b_norm=b_norm, affine=affine, dropout=dropout)
        self.convr2 = Gated_block(out_channels, out_channels, kernel_size=(3, 3), stride=1, padding=padding, b_norm=b_norm, affine=affine, dropout=dropout)

    def _crop_concat(self, upsampled, downsampled):
        """
         pad upsampled to the (h, w) of downsampled to concatenate
         for expansive path.
        Returns:
            The concatenated tensor
        """
        h = downsampled.size()[2] - upsampled.size()[2]
        w = downsampled.size()[3] - upsampled.size()[3]
        upsampled = F.pad(upsampled, (0, w, 0, h))
        return torch.cat((downsampled, upsampled), 1)

    def forward(self, x, down_tensor):
        x = self.upSample(x)
        #print(x.shape)
        #print(down_tensor.shape)
        if self.bilinear:
            x = self.conv(x)    # extra conv, will remove later
        x = self._crop_concat(x, down_tensor)
        x = self.convr1(x)
        x = self.convr2(x)
        return x


class Gated_UNet(nn.Module):
    def __init__(self, n_channels=16, input_channels=1, bnorm=False, affine=False, bilinear=False, dropout=None):
        super(Gated_UNet, self).__init__()
        self.down1 = Encoder_block_gated(input_channels, n_channels, bnorm, affine, dropout=dropout)
        self.down2 = Encoder_block_gated(n_channels, n_channels*2, bnorm, affine, dropout=dropout)
        self.down3 = Encoder_block_gated(n_channels*2, n_channels*4, bnorm, affine, dropout=dropout)
        self.down4 = Encoder_block_gated(n_channels*4, n_channels*8, bnorm, affine, dropout=dropout)

        self.center = nn.Sequential(
            Gated_block(n_channels*8, n_channels*16, kernel_size=(3, 3), stride=1, padding=1, b_norm=bnorm, affine=affine, dropout=dropout),
            Gated_block(n_channels*16, n_channels*16, kernel_size=(3, 3), stride=1, padding=1, b_norm=bnorm, affine=affine, dropout=dropout)
        )

        self.up1 = Decoder_block_gated(in_channels=n_channels*16, out_channels=n_channels*8, upsample_factor=2, padding = 1, b_norm=bnorm, affine=affine, bilinear=bilinear, dropout=dropout)
        self.up2 = Decoder_block_gated(in_channels=n_channels*8, out_channels=n_channels*4, upsample_factor=2, padding = 1, b_norm=bnorm, affine=affine, bilinear=bilinear, dropout=dropout)
        self.up3 = Decoder_block_gated(in_channels=n_channels*4, out_channels=n_channels*2, upsample_factor=2, padding = 1, b_norm=bnorm, affine=affine, bilinear=bilinear, dropout=dropout)
        self.up4 = Decoder_block_gated(in_channels=n_channels*2, out_channels=n_channels, upsample_factor=2, padding = 1, b_norm=bnorm, affine=affine, bilinear=bilinear, dropout=dropout)

        # 1x1 convolution at the last layer
        self.outputNN = nn.Conv2d(n_channels, 1, kernel_size=(1, 1), padding=0, stride=1)

        self._initialize_weights()

    def forward(self, x):
        #x1 = x
        x, skip1 = self.down1(x)
        x, skip2 = self.down2(x)
        x, skip3 = self.down3(x)
        x, skip4 = self.down4(x)
        x = self.center(x)
        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)
        x = self.outputNN(x)
        #x = x + x1
        x = torch.sigmoid(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

