import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import torch.nn.functional as F

num_classes = 64

zht_output = 3

class alexnet_siamese(nn.Module):

	def __init__(self,cwd):
		super(alexnet_siamese, self).__init__()

		pa1 = torch.load(cwd+"/../../networks/models/alexnet_places365.pth.tar",map_location=torch.device('cpu'))
		pa2 = torch.load(cwd+"/../../networks/models/alexnet_places365.pth.tar",map_location=torch.device('cpu'))

		alex_conv1 = nn.Sequential(*list(pa1.children())[0])
		alex_conv2 = nn.Sequential(*list(pa2.children())[0])
		alex_average_pool1 = list(pa1.children())[1]
		alex_average_pool2 = list(pa2.children())[1]


		self.GE_conv = alex_conv1
		self.GM_conv = alex_conv2
		self.GE_average = alex_average_pool1
		self.GM_average = alex_average_pool2

		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(2 * 256 * 6 * 6, 4096),
			nn.LeakyReLU(negative_slope=0.01, inplace=True),
			nn.Dropout(),
			nn.Linear(4096, num_classes),
			# nn.Softmax()
		)

		self.zht = nn.Sequential(
			nn.Dropout(),
			nn.Linear(2 * 256 * 6 * 6, 4096),
			nn.LeakyReLU(negative_slope=0.01, inplace=True),
			nn.Dropout(),
			nn.Linear(4096, zht_output),
			nn.Tanh(),
		)

	def forward(self, ge, gm):

		xe = self.GE_conv(ge)
		xm = self.GM_conv(gm)

		xe = self.GE_average(xe)
		xm = self.GM_average(xm)

		xe = xe.view(xe.size(0), 256 * 6 * 6)
		xm = xm.view(xm.size(0), 256 * 6 * 6)

		fx = torch.cat((xe, xm),1)

		xy_classes = self.classifier(fx)
		zht = self.zht(fx)

		#output = torch.cat((xy_classes, zht),1)

		return xy_classes,zht
