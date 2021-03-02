import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch.nn.functional as F


class scene_network(nn.Module):

    def __init__(self,cwd,model):
        super(scene_network, self).__init__()

        self.model = model

        pa1 = torch.load(cwd+"/../../networks/models/"+self.model+"_places365.pth.tar",map_location=torch.device('cpu'))
        pa2 = torch.load(cwd+"/../../networks/models/"+self.model+"_places365.pth.tar",map_location=torch.device('cpu'))


        self.resnet_1 = nn.Sequential(*(list(pa1.children())[:-1]))
        self.resnet_2 = nn.Sequential(*(list(pa2.children())[:-1]))

    def forward(self,ge,gm):

        features_1 = self.resnet_1(ge)
        features_2 = self.resnet_2(ge)

        return F.pairwise_distance(features_1, features_2)
