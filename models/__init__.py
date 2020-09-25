import torch
import torch.nn as nn
import torchvision.models as models

from .UNet import UNet_Model
from .refiner_network import Refiner_Network

def convertBNtoGN(module, num_groups=16):
  if isinstance(module, torch.nn.modules.batchnorm.BatchNorm2d):
    return nn.GroupNorm(num_groups, module.num_features,
                        eps=module.eps, affine=module.affine)
    if module.affine:
        mod.weight.data = module.weight.data.clone().detach()
        mod.bias.data = module.bias.data.clone().detach()

  for name, child in module.named_children():
      module.add_module(name, convertBNtoGN(child, num_groups=num_groups))

  return module


def get_models(model_name):
    model_name = model_name.lower() # handle character case error

    if model_name == 'unet':
        model = UNet_Model(n_channels=2, n_classes=2)
    elif model_name == 'fcn':
        model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=2, pretrained_backbone=False)
        model.backbone.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False)
        model = convertBNtoGN(model)
    elif model_name == 'refiner':
        model = Refiner_Network(UNet_Model)
    else:
        print('FATAL: Invalid model name "{}"'.format(model_name))
        exit()
    return model
                
