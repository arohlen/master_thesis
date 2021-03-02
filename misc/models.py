import torch
import torchvision.models as models
import cv2
from PIL import Image
from torchvision import datasets, models, transforms
import torch.nn as nn
from scene_network_resnet import scene_network
from camera_network_resnet import camera_network

# path = "D:\Shetty_data"

# camera_localization_model_path = path+"\models\camera_localization_hybrid_network.pth.tar"
# scene_localization_model_path = path+"\models\scene_localization_network.pth.tar"

# scene_model = torch.load(scene_localization_model_path,map_location=torch.device('cpu'))
# camera_model = torch.load(camera_localization_model_path,map_location=torch.device('cpu'))

# for key in scene_model:
#     print(key)

# print(scene_model["optimizer"])
# print()

# for key in scene_model["optimizer"]["state"]:
#     print(key)

# print(scene_model["optimizer"]["state"][140438310971288])

# pa1 = torch.hub.load('pytorch/vision:v0.6.0', 'alexnet')

# print("Model's state_dict:")
# for param_tensor in pa1.state_dict():
#     print(param_tensor, "\t", pa1.state_dict()[param_tensor].size())

# model = torch.load("alexnet_places365.pth.tar",map_location=torch.device('cpu'))

# print("Model's state_dict:")
# for param_tensor in model["state_dict"]:
#     print(param_tensor, "\t", model["state_dict"][param_tensor].size())


# model1 = torch.load("resnet50_places365.pth.tar",map_location=torch.device('cpu'))

# print("Model1's state_dict:")
# for param_tensor in model1["state_dict"]:
#     print(param_tensor, "\t", model1["state_dict"][param_tensor].size())


# for key in model1["state_dict"]:
#     print(key)


# print(model1["arch"])

# model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet50')

# print(model)

# # model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18')

model = models.__dict__["alexnet"](num_classes=365)

# # print("Model's state_dict:")
# # for param_tensor in model.state_dict():
# #     print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# # print(model)

# res50 = torch.load("alexnet_places365.pth.tar",map_location=torch.device('cpu'))

# print()
# state_dict = {}
# # print("res18's state_dict:")
# for param_tensor in res50["state_dict"]:
#     # print(param_tensor, "\t", res18["state_dict"][param_tensor].size())
#     state_dict[param_tensor.replace("module.","")] = res50["state_dict"][param_tensor]
#     # print(param_tensor.replace("module.",""),"       ",param_tensor)
# # # for key in state_dict:
# # #     print(key)

# model.load_state_dict(state_dict)

# # # # print(model)

# torch.save(model,"alexnet_365.pth.tar")


# model = torch.load("res18_365.pth.tar",map_location=torch.device('cpu'))

# print(res18)
# model = nn.Sequential(*(list(model.children())[:-1]))

model = scene_network(model="resnet18")

model.eval()

model2 = camera_network(model="resnet18")

model2.eval()

uav_img = cv2.imread("D:\Shetty_data\\train\\atlanta_austin\\atlanta\\atlanta_uav\\uav\\uav0.png")
sat_img = cv2.imread("D:\Shetty_data\\train\\atlanta_austin\\atlanta\\atlanta_sat\sat300\sat0.png")

# uav_img = cv2.resize(uav_img, (224, 224))
# sat_img = cv2.resize(sat_img, (224, 224))

# cv2.imshow('image',uav_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

uav_img = Image.fromarray(uav_img).convert("RGB")
sat_img = Image.fromarray(sat_img).convert("RGB")

to_tensor = transforms.ToTensor()

uav_img = to_tensor(uav_img)
sat_img = to_tensor(sat_img)
uav_img_batch = uav_img.unsqueeze(0)
sat_img_batch = sat_img.unsqueeze(0)


print(model(uav_img_batch,sat_img_batch))
print(model2(uav_img_batch,sat_img_batch).size())
