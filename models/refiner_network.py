import torch
import torch.nn as nn
from UNet import UNet_Model

class Refiner_Network(nn.Module):
    def __init__(self, seg_model_class):
        super(Refiner_Network, self).__init__()
        # assumes that seg_model_class have the same input reqs
        self.stage_1  = seg_model_class(n_channels=3, n_classes=2)
        self.stage_2  = seg_model_class(n_channels=5, n_classes=2)

        # self.softmax = nn.Softmax(dim=1)

    def forward(self, batch):
        init_pred = self.stage_1(batch)  # assume stage_1 is a regular sem seg network
        # init_pred = self.softmax(init_pred)

        # combine input data and initial_prediction
        combined_input = torch.cat((batch, init_pred), dim=1)  # combine over the channel dim.

        refined_pred = self.stage_2(combined_input)
        # refined_pred = self.softmax(refined_pred)

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