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


def get_model(model_name, dataset_name, cs_points=None):
    model_name = model_name.lower() # handle character case error
    dataset_name = dataset_name.lower()

    if dataset_name[:4] == 'sen1':
        n_channels = 2
    elif dataset_name[:4] == 'sen2':
        n_channels = 13
    else:
        print('FATAL: Invalid dataset name "{}"'.format(dataset_name))
        exit()

    if model_name == 'unet':
        model = UNet_Model(n_channels=n_channels, n_classes=2)
    elif model_name == 'fcn':
        model = models.segmentation.fcn_resnet50(pretrained=False, num_classes=2, pretrained_backbone=False)
        model.backbone.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3,
                                         bias=False)
        model = convertBNtoGN(model)
    elif model_name == 'refiner':
        model = UNet_Model(n_channels=n_channels, n_classes=2)
        model = Refiner_Network(UNet_Model, dataset_name, cs_points)
    else:
        print('FATAL: Invalid model name "{}"'.format(model_name))
        exit()
    return model