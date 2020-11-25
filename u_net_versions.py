import torch
import torch.nn as nn
import torch.nn.functional as F


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

