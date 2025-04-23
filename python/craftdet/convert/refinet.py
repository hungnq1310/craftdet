import torch
import torch.nn as nn



class RefineNet(nn.Module):
  """
  Implementation of refine network.
  """

  def __init__(self):
    """
    CLass constructor.
    """
    super().__init__()
    # last convolution layer.
    self.last_conv = nn.Sequential(
      nn.Conv2d(34, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
      nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
      nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True)
    )
    # refine aspp layer 1.
    self.aspp1 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, dilation=6, padding=6), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
      nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
      nn.Conv2d(128, 1, kernel_size=1)
    )
    # refine aspp layer 2.
    self.aspp2 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, dilation=12, padding=12), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
      nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
      nn.Conv2d(128, 1, kernel_size=1)
    )
    # refine aspp layer 3.
    self.aspp3 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, dilation=18, padding=18), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
      nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
      nn.Conv2d(128, 1, kernel_size=1)
    )
    # refine aspp layer 4.
    self.aspp4 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=3, dilation=24, padding=24), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
      nn.Conv2d(128, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
      nn.Conv2d(128, 1, kernel_size=1)
    )
    # init weights.
    init_weights(self.last_conv.modules())
    init_weights(self.aspp1.modules())
    init_weights(self.aspp2.modules())
    init_weights(self.aspp3.modules())
    init_weights(self.aspp4.modules())

  def forward(self, y, upconv4):
    """
    Forward function.
    :param x:   inputs of network.
    :return:    outputs of network.
    """
    refine = torch.cat([y.permute(0,3,1,2), upconv4], dim=1)
    refine = self.last_conv(refine)
    aspp1 = self.aspp1(refine)
    aspp2 = self.aspp2(refine)
    aspp3 = self.aspp3(refine)
    aspp4 = self.aspp4(refine)
    #out = torch.add([aspp1, aspp2, aspp3, aspp4], dim=1)
    out = aspp1 + aspp2 + aspp3 + aspp4
    return out.permute(0, 2, 3, 1)  # , refine.permute(0,2,3,1)
import torch
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


# --------------------------------------------------------CONVERT-------------------------------------------
import os
import torch

craft_net = RefineNet()

IMAGE_SIZE = int(os.getenv('IMAGE_SIZE', default=512))
craft = '/kaggle/input/weight-craft/weights/craft/refinerCTW1500.pth'
use_cuda = torch.cuda.is_available()

state_dict = torch.load(craft, map_location='cuda' if use_cuda else 'cpu')
if 'module.' in list(state_dict.keys())[0]:
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

craft_net.load_state_dict(state_dict)
if use_cuda:
    craft_net = craft_net.cuda()

torch_input1 = torch.randn((1, IMAGE_SIZE, IMAGE_SIZE, 2), device='cuda' if use_cuda else 'cpu')
torch_input2 = torch.randn((1, 32, 512, 512), device='cuda' if use_cuda else 'cpu')

onnx_path = "/kaggle/working/refinet_onnx.onnx"

torch.onnx.export(
    craft_net,
    (torch_input1, torch_input2),
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
# -----------------------------------------------------------------------------------------------------------