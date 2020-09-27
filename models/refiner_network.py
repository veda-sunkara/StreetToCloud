import torch
import torch.nn as nn
from .UNet import UNet_Model

class Refiner_Network(nn.Module):
    def __init__(self, seg_model_class, dataset_name, cs_points=None):
        super(Refiner_Network, self).__init__()
        self.cs_points = cs_points

        if dataset_name[:4] == 'sen1':
            n_channels = 2
        elif dataset_name[:4] == 'sen2':
            n_channels = 13
        
        # assumes that seg_model_class have the same input reqs
        self.stage_1  = seg_model_class(n_channels=n_channels, n_classes=2)

        if cs_points:
            self.stage_2  = seg_model_class(n_channels=n_channels+3, n_classes=2)
        else:
            self.stage_2  = seg_model_class(n_channels=n_channels+2, n_classes=2)

        # self.softmax = nn.Softmax(dim=1)

    def forward(self, batch):
        if self.cs_points:
            img, cs_points = batch
        else:
            img = batch

        init_pred = self.stage_1(img)  # assume stage_1 is a regular sem seg network

        # combine input data and initial_prediction
        if self.cs_points:
            combined_input = torch.cat((img, init_pred, cs_points), dim=1)  # combine over the channel dim.
        else:
            combined_input = torch.cat((img, init_pred), dim=1)  # combine over the channel dim.

        refined_pred = self.stage_2(combined_input)

        return refined_pred


def get_refiner_network(seg_model_class):
    return Refiner_Network(seg_model_class)

if __name__ == '__main__':
    # put test input through the network
    device = torch.device('cuda')

    net = Refiner_Network(UNet_Model).to(device)
    input = torch.ones((2, 3, 512, 512)).to(device)
    output = net(input)
    print(output.shape)  # should be [2, 2, 512, 512]