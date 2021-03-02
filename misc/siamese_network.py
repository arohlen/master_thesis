import torch
from torchvision import datasets, models, transforms
import numpy as np
import cv2
from PIL import Image

from scene_network_alexnet import alexnet_siamese as scene_network
from camera_network_alexnet import alexnet_siamese as camera_network

scene_model = scene_network()
camera_model = camera_network()

scene_model.eval()
camera_model.eval()

path = "D:\Shetty_data"

# scene_localization_model_path = path+"\models\scene_localization_network.pth.tar"
# camera_localization_model_path = path+"\models\camera_localization_hybrid_network.pth.tar"

# scene_model.load_state_dict(torch.load(scene_localization_model_path,map_location=torch.device('cpu'))["state_dict"])
# camera_model.load_state_dict(torch.load(camera_localization_model_path,map_location=torch.device('cpu'))["state_dict"])


uav_img = cv2.imread("D:\Shetty_data\\train\\atlanta\\uav\\uav0.png")
sat_img = cv2.imread("D:\Shetty_data\\train\\atlanta\\sat300\sat0.png")

uav_img = Image.fromarray(uav_img).convert("RGB")
sat_img = Image.fromarray(sat_img).convert("RGB")

to_tensor = transforms.ToTensor()

uav_img = to_tensor(uav_img)
sat_img = to_tensor(sat_img)
uav_img_batch = uav_img.unsqueeze(0)
sat_img_batch = sat_img.unsqueeze(0)


output1 = scene_model(uav_img_batch,sat_img_batch)
xy,zHTR = camera_model(uav_img_batch,sat_img_batch)

print(output1)
print(torch.max(xy))
print(zHTR)

