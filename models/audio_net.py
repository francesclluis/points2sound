
import torch
import torch.nn as nn
import math


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference)**0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


def center_trim(tensor, reference):
    if hasattr(reference, "size"):
        reference = reference.size(-1)
    diff = tensor.size(-1) - reference
    if diff < 0:
        raise ValueError("tensor must be larger than reference")
    if diff:
        tensor = tensor[..., diff // 2:-(diff - diff // 2)]
    return tensor


class Demucs(nn.Module):
    def __init__(
            self,
            n_audio_channels: int = 1,  # pylint: disable=redefined-outer-name
            kernel_size: int = 8,
            stride: int = 4,
            visual_feature_size: int = 16,
            context: int = 3,
            depth: int = 6,
            channels: int = 64,
            growth: float = 2.0,
            lstm_layers: int = 2,
            rescale: float = 0.1):  # pylint: disable=redefined-outer-name
        super().__init__()
        self.n_audio_channels = n_audio_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.visual_feature_size = visual_feature_size
        self.context = context
        self.depth = depth
        self.channels = channels
        self.growth = growth
        self.lstm_layers = lstm_layers
        self.rescale = rescale

        self.encoder = nn.ModuleList()  # Source encoder
        self.decoder = nn.ModuleList()  # Audio output decoder

        activation = nn.GLU(dim=1)

        in_channels = n_audio_channels  # Number of input channels

        # Wave U-Net structure
        for index in range(depth):
            encode = nn.ModuleDict()
            encode["conv1"] = nn.Conv1d(in_channels, channels, kernel_size,
                                        stride)
            encode["relu"] = nn.ReLU()

            encode["conv2"] = nn.Conv1d(channels, 2 * channels, 1)
            encode["activation"] = activation

            encode["gc_embed1"] = nn.Conv1d(self.visual_feature_size, channels, 1)
            encode["gc_embed2"] = nn.Conv1d(self.visual_feature_size, 2 * channels, 1)

            self.encoder.append(encode)

            decode = nn.ModuleDict()
            if index > 0:
                out_channels = in_channels
            else:
                out_channels = 2 * n_audio_channels

            decode["conv1"] = nn.Conv1d(channels, 2 * channels, context)
            decode["activation"] = activation
            decode["conv2"] = nn.ConvTranspose1d(channels, out_channels,
                                                 kernel_size, stride)

            decode["gc_embed1"] = nn.Conv1d(self.visual_feature_size, 2 * channels, 1)
            decode["gc_embed2"] = nn.Conv1d(self.visual_feature_size, out_channels, 1)

            if index > 0:
                decode["relu"] = nn.ReLU()
            self.decoder.insert(0,
                                decode)  # Put it at the front, reverse order

            in_channels = channels
            channels = int(growth * channels)

        # Bi-directional LSTM for the bottleneck layer
        channels = in_channels
        self.lstm = nn.LSTM(bidirectional=True,
                            num_layers=lstm_layers,
                            hidden_size=channels,
                            input_size=channels)
        self.lstm_linear = nn.Linear(2 * channels, channels)

        rescale_module(self, reference=rescale)

    def forward(self, mono_mix, visual_feature):

        x = mono_mix
        saved = [x]

        # Encoder
        for encode in self.encoder:
            x = encode["conv1"](x)  # Conv 1d
            embedding = encode["gc_embed1"](visual_feature.unsqueeze(2))

            x = encode["relu"](x + embedding)
            x = encode["conv2"](x)

            embedding2 = encode["gc_embed2"](visual_feature.unsqueeze(2))
            x = encode["activation"](x + embedding2)
            saved.append(x)

        # Bi-directional LSTM at the bottleneck layer
        x = x.permute(2, 0, 1)  # prep input for LSTM
        self.lstm.flatten_parameters()  # to improve memory usage.
        x = self.lstm(x)[0]
        x = self.lstm_linear(x)
        x = x.permute(1, 2, 0)

        # Source decoder
        for decode in self.decoder:
            skip = center_trim(saved.pop(-1), x)
            x = x + skip

            x = decode["conv1"](x)
            embedding = decode["gc_embed1"](visual_feature.unsqueeze(2))
            x = decode["activation"](x + embedding)
            x = decode["conv2"](x)
            embedding2 = decode["gc_embed2"](visual_feature.unsqueeze(2))
            if "relu" in decode:
                x = decode["relu"](x + embedding2)

        # Reformat the output
        x = x.view(x.size(0), 2, self.n_audio_channels, x.size(-1))

        return x

    def valid_length(self, length: int) -> int:

        for _ in range(self.depth):
            length = math.ceil((length - self.kernel_size) / self.stride) + 1
            length = max(1, length)
            length += self.context - 1

        for _ in range(self.depth):
            length = (length - 1) * self.stride + self.kernel_size

        return int(length)


def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])


def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])


def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))
    if(Relu):
        model.append(nn.ReLU())
    return nn.Sequential(*model)


class UNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2, visual_feature_size=64):
        super(UNet, self).__init__()

        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_upconvlayer1 = unet_upconv(512 + visual_feature_size, ngf * 8)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf *4)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 2, output_nc, True)

    def forward(self, x, visual_feature):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)

        visual_feature = visual_feature.view(visual_feature.shape[0], -1, 1, 1)  # flatten visual feature
        visual_feature = visual_feature.repeat(1, 1, audio_conv5feature.shape[-2], audio_conv5feature.shape[-1])  # tile visual feature

        audioVisual_feature = torch.cat((visual_feature, audio_conv5feature), dim=1)

        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv4feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv3feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv2feature), dim=1))
        mask_prediction = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv1feature), dim=1)) * 2 - 1
        return mask_prediction
