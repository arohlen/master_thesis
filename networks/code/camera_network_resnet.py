import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch.nn.functional as F



class camera_network(nn.Module):

    def __init__(self,cwd,model):
        super(camera_network, self).__init__()

        self.model = model

        pa1 = torch.load(cwd+"/../../networks/models/"+self.model+"_places365.pth.tar",map_location=torch.device('cpu'))
        pa2 = torch.load(cwd+"/../../networks/models/"+self.model+"_places365.pth.tar",map_location=torch.device('cpu'))

        self.resnet_1 = nn.Sequential(*(list(pa1.children())[:-1]))
        self.resnet_2 = nn.Sequential(*(list(pa2.children())[:-1]))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048*2, 4096) if self.model == "resnet50" else nn.Linear(512*2, 4096),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 64),
            nn.Softmax()
        )

        self.zht = nn.Sequential(
            nn.Dropout(),
            nn.Linear(2048*2, 4096) if self.model == "resnet50" else nn.Linear(512*2, 4096),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 3),
        )



    def forward(self,ge,gm):

        features_1 = self.resnet_1(ge)
        features_2 = self.resnet_2(ge)

        features = torch.cat((features_1, features_2),1)

        features = features.view(features.size(0),features.size(1))

        xy_classes = self.classifier(features)
        zht = self.zht(features)

        #output = torch.cat((xy_classes, zht),1)

        return xy_classes,zht
