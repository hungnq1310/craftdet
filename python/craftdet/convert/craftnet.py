import torch
from torchvision import models
from collections import namedtuple


class BaseNet(torch.nn.Module):
    """
    CRAFT base network is based on VGG16 backbone.
    """

    def __init__(
            self,
            freeze: bool = True,
            pretrained: bool = True,
            weights: str = "VGG16_BN_Weights.IMAGENET1K_V1"):
        """
        Create base network.
        :param freeze:      freeze the first convolution layer.
        :param pretrained:  use pretrained weights or not.
        :param weights:     pretrained weights to be used.
        """
        super().__init__()
        # make vgg16 features.
        features = models.vgg16_bn(weights=weights).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(12):  # conv2_2
            self.slice1.add_module(str(x), features[x])
        for x in range(12, 19):  # conv3_3
            self.slice2.add_module(str(x), features[x])
        for x in range(19, 29):  # conv4_3
            self.slice3.add_module(str(x), features[x])
        for x in range(29, 39):  # conv5_3
            self.slice4.add_module(str(x), features[x])
        # fc6, fc7 without atrous conv
        self.slice5 = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            torch.nn.Conv2d(1024, 1024, kernel_size=1)
        )
        # init weights.
        if not pretrained:
            init_weights(self.slice1.modules())
            init_weights(self.slice2.modules())
            init_weights(self.slice3.modules())
            init_weights(self.slice4.modules())
        # no pretrained model for fc6 and fc7.
        init_weights(self.slice5.modules())
        if freeze:
            for param in self.slice1.parameters():  # only first conv
                param.requires_grad = False

    def forward(self, x):
        """
        Forward function.
        :param x:   inputs of network.
        :return:    outputs of network.
        """
        h = self.slice1(x)
        h_relu2_2 = h
        h = self.slice2(h)
        h_relu3_2 = h
        h = self.slice3(h)
        h_relu4_3 = h
        h = self.slice4(h)
        h_relu5_3 = h
        h = self.slice5(h)
        h_fc7 = h
        vgg_outputs = namedtuple("VggOutputs", ['fc7', 'relu5_3', 'relu4_3', 'relu3_2', 'relu2_2'])
        return vgg_outputs(h_fc7, h_relu5_3, h_relu4_3, h_relu3_2, h_relu2_2)


import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double of Conv, Batchnorm, ReLU blocks.
    """

    def __init__(self, in_ch, mid_ch, out_ch) -> None:
        """
        Class constructor.
        :param in_ch:   input channels.
        :param mid_ch:  middle channels.
        :param out_ch:  output channels.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + mid_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward function.
        :param x:   inputs of network.
        :return:    outputs of network.
        """
        return self.conv(x)


class CraftNet(nn.Module):
    """
    Implementation of CRAFT network.
    """

    def __init__(self, pretrained=False, freeze=False):
        super().__init__()
        # base network.
        self.basenet = BaseNet(pretrained=pretrained, freeze=freeze)
        # U-network.
        self.upconv1 = DoubleConv(1024, 512, 256)
        self.upconv2 = DoubleConv(512, 256, 128)
        self.upconv3 = DoubleConv(256, 128, 64)
        self.upconv4 = DoubleConv(128, 64, 32)
        # final cls.
        num_class = 2
        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, num_class, kernel_size=1),
        )
        # init weights.
        init_weights(self.upconv1.modules())
        init_weights(self.upconv2.modules())
        init_weights(self.upconv3.modules())
        init_weights(self.upconv4.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        """
        Forward function.
        :param x:   inputs of network.
        :return:    outputs of network.
        """
        sources = self.basenet(x)
        # U-network.
        y = torch.cat([sources[0], sources[1]], dim=1)
        y = self.upconv1(y)
        # layer 2.
        y = F.interpolate(y, size=sources[2].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[2]], dim=1)
        y = self.upconv2(y)
        # layer 3.
        y = F.interpolate(y, size=sources[3].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[3]], dim=1)
        y = self.upconv3(y)
        # layer 4.
        y = F.interpolate(y, size=sources[4].size()[2:], mode='bilinear', align_corners=False)
        y = torch.cat([y, sources[4]], dim=1)
        feature = self.upconv4(y)
        # final cls.
        y = self.conv_cls(feature)
        # return permute results.
        return y.permute(0, 2, 3, 1), feature
import torch
import os
import torch.nn as nn
from pathlib import Path
from collections import OrderedDict


def init_weights(modules):
  """
  Initial weights of modules.
  :param modules:   modules to be initialized.
  """
  for m in modules:
    if isinstance(m, nn.Conv2d):
      torch.nn.init.xavier_uniform_(m.weight.data)
      if m.bias is not None:
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
      m.weight.data.fill_(1)
      m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
      m.weight.data.normal_(0, 0.01)
      m.bias.data.zero_()


def copy_state_dict(state_dict):
    """
    Copy state dict into new one that fit craft networks.
    :param state_dict:  input state dict.
    :return:            fitted state dict.
    """
    if list(state_dict.keys())[0].startswith("module"):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = ".".join(k.split(".")[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def load_model(model, weight_path: str, cuda: bool = True):
  """
  Load weight into model.
  :param model:       target model.
  :param weight_path: path to weight file.
  :param cuda:        use cuda or not.
  :return:            model with new weight.
  """
  weight_path = Path(weight_path).expanduser().resolve()
  weight_path.parent.mkdir(exist_ok=True, parents=True)
  weight_path = str(weight_path)
  # load model into device.
  if torch.cuda.is_available() and cuda:
    model.load_state_dict(copy_state_dict(torch.load(weight_path)))
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = False
  else:
    model.load_state_dict(copy_state_dict(torch.load(weight_path, map_location='cpu')))
  model.eval()
  return model

#-----------------------CONVERT ---------------------------------
import os
import torch

craft_net = CraftNet()

IMAGE_SIZE = int(os.getenv('IMAGE_SIZE', default=512))
craft = '/kaggle/input/weight-craft/weights/craft/mlt25k.pth'
use_cuda = torch.cuda.is_available()

state_dict = torch.load(craft, map_location='cuda' if use_cuda else 'cpu')
if 'module.' in list(state_dict.keys())[0]:
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

craft_net.load_state_dict(state_dict)
if use_cuda:
    craft_net = craft_net.cuda()

torch_input = torch.randn((1, 3, IMAGE_SIZE, IMAGE_SIZE), device='cuda' if use_cuda else 'cpu')

onnx_path = "/kaggle/working/craftnet.onnx"

torch.onnx.export(
    craft_net,
    torch_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'},
                  'output': {0: 'batch_size'}}
)

print("CraftNet model exported successfully to ONNX.")
# -------------------------------------------------------------------------