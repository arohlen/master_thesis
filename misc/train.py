import pickle

with open('D:/Shetty_data/train/uav_image_data.pickle', 'rb') as handle:
    matching_uav_data = pickle.load(handle)

with open('D:/Shetty_data/train/uav_image_data.pickle', 'rb') as handle:
    non_matching_uav_data = pickle.load(handle)

with open('D:/Shetty_data/train/matching_sat_data.pickle', 'rb') as handle:
    matching_sat_data = pickle.load(handle)

with open('D:/Shetty_data/train/non_matching_sat_data.pickle', 'rb') as handle:
    non_matching_sat_data = pickle.load(handle)


import numpy as np

def contrastive_loss(d,l):
    m = 100
    loss = l*(d**2) + (1-l)*np.maximum(0,m-d.detach().numpy())**2
    return loss

import torch

matching_uav_training = torch.stack(matching_uav_data)
non_matching_uav_training = torch.stack(non_matching_uav_data)
matching_sat_training = torch.stack(matching_sat_data)
non_matching_sat_training = torch.stack(non_matching_sat_data)


from torchvision import datasets, models, transforms
from scene_network_alexnet import alexnet_siamese as scene_network
import torch.optim as optim
from sklearn.utils import shuffle

scene_model = scene_network()
optimizer = optim.SGD(scene_model.parameters(), lr=10e-5)


epochs = 10
batch_size = 10

for epoch in range(epochs):

    matching_uav_training,matching_sat_training = shuffle(matching_uav_training,matching_sat_training)

    non_matching_uav_training,non_matching_sat_training = shuffle(non_matching_uav_training,non_matching_sat_training)

    running_loss = 0.0

    for i in range(len(matching_sat_training)//batch_size):

        uav_input = torch.cat((matching_uav_training[i*(batch_size//2):(i+1)*(batch_size//2)],non_matching_uav_training[i*(batch_size//2):(i+1)*(batch_size//2)]))

        sat_input = torch.cat((matching_sat_training[i*(batch_size//2):(i+1)*(batch_size//2)],non_matching_sat_training[i*(batch_size//2):(i+1)*(batch_size//2)]))


        optimizer.zero_grad()

        labels = torch.tensor(([1]*(batch_size//2))+[0]*(batch_size//2))

        distances = scene_model(uav_input,sat_input)

        print(distances)

        loss = contrastive_loss(distances, labels)

        loss.sum().backward()

        optimizer.step()

        running_loss += loss.sum().item()

        if i % 5 == 4:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 5))
            running_loss = 0.0