import h5py
import pickle5 as pickle
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import torch
from tqdm import tqdm
import torch.optim as optim
from sklearn.utils import shuffle
import numpy as np

# Get data paths

path = "/local_storage/datasets/shetty_gao/"

# Get data labels

with open(path+'training_labels.pickle', 'rb') as handle:
    labels = pickle.load(handle)

uav_training_paths = labels["uav_training_paths"]
sat_training_paths = labels["sat_training_paths"]

uav_validation_paths = labels["uav_validation_paths"]
sat_validation_paths = labels["sat_validation_paths"]

grid_labels_training = labels["grid_labels_training"]
grid_labels_validation = labels["grid_labels_validation"]

# Create weighting for loss function

cuda = torch.device('cuda')

occurances = np.zeros(64,dtype="float32")

for i in range(len(grid_labels_training)):
    occurances[grid_labels_training[i]] += 1

weights = 1/occurances
weights /= weights.max()
weights = torch.tensor(weights).cuda()

#Camera network class

import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F

num_classes = 64

zht_output = 1

class camera_network(nn.Module):

    def __init__(self,cwd):
        super(camera_network, self).__init__()

        pa1 = torch.load(cwd+"alexnet_places365.pth.tar")
        pa2 = torch.load(cwd+"alexnet_places365.pth.tar")

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
        )

        self.z = nn.Sequential(
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
        z = self.z(fx)

        output = torch.cat((xy_classes, z),1)

        return output


def loss_func(output,grid_labels):

    cross_entropy = nn.CrossEntropyLoss(weight=weights)

    alpha = 30
    loss_grid = cross_entropy(output[:,:64],grid_labels)
    loss = alpha*loss_grid

    return loss

# Function for getting data

def get_data_augmentation(uav_paths,sat_paths):

    train_path = path+'train_resized/'

    cuda = torch.device('cuda')

    uav_images = []
    sat_images = []

    for i in range(len(uav_paths)):

        if uav_paths[i][-2] == "_":
            angle = 90 * int(uav_paths[i][-1])

            uav_path = train_path+uav_paths[i][:-2]+".png"
            sat_path = train_path+sat_paths[i][:-2]+".png"

            sat_img = Image.open(sat_path).convert("RGB")
            sat_img = sat_img.rotate(angle)

        else:

            uav_path = train_path+uav_paths[i]
            sat_path = train_path+sat_paths[i]

            sat_img = Image.open(sat_path).convert("RGB")


        uav_img = Image.open(uav_path).convert("RGB")

        to_tensor = transforms.ToTensor()

        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        uav_tensor = normalize(to_tensor(uav_img)).cuda()
        sat_tensor = normalize(to_tensor(sat_img)).cuda()

        uav_images.append(uav_tensor)
        sat_images.append(sat_tensor)

    return torch.stack(uav_images),torch.stack(sat_images)

# Train grid part of model
import matplotlib.pyplot as plt
learning_rate = 3*10e-5

camera_model = camera_network(path)
optimizer = optim.SGD(camera_model.parameters(), lr=learning_rate)

cuda = torch.device('cuda')

grid_labels_training = torch.tensor(grid_labels_training).cuda()
grid_labels_validation = torch.tensor(grid_labels_validation).cuda()

camera_model.cuda()

camera_model.train()

validation_acc = []
training_acc = []

max_val_acc = 0

epochs = 20
batch_size = 64

for epoch in range(epochs):

    uav_training_paths,sat_training_paths,grid_labels_training = shuffle(uav_training_paths,sat_training_paths,grid_labels_training)

    pbar = tqdm(range(len(uav_training_paths)//batch_size),desc="Epoch: {}".format(epoch))

    training_acc_sum = 0
    
    for i in pbar:

        uav_input,sat_input = get_data_augmentation(uav_training_paths[i*(batch_size):(i+1)*(batch_size)],sat_training_paths[i*(batch_size):(i+1)*(batch_size)])

        grid_labels = grid_labels_training[i*(batch_size):(i+1)*(batch_size)]

        optimizer.zero_grad()

        output = camera_model(uav_input,sat_input)

        loss = loss_func(output,grid_labels)

        acc = np.mean((torch.argmax(output[:,:64],1) == grid_labels).cpu().detach().numpy())

        training_acc_sum += acc

        pbar.set_description("Epoch: {} Loss: {} Acc: {}".format(epoch,loss.item(),acc))

        loss.backward()

        optimizer.step()

    
    training_acc.append(training_acc_sum/(len(uav_training_paths)//batch_size))

    val_loss_sum = 0
    acc_sum = 0

    for i in tqdm(range(len(uav_validation_paths)//batch_size),desc="Calculating validation loss"):

        uav_input,sat_input = get_data_augmentation(uav_validation_paths[i*(batch_size):(i+1)*(batch_size)],sat_validation_paths[i*(batch_size):(i+1)*(batch_size)])

        grid_labels = grid_labels_validation[i*(batch_size):(i+1)*(batch_size)]

        output = camera_model(uav_input,sat_input)
        val_loss = loss_func(output,grid_labels)

        acc = np.mean((torch.argmax(output[:,:64],1) == grid_labels).cpu().detach().numpy())
        acc_sum += acc

        val_loss_sum += val_loss.detach().item()

    validation_acc.append(acc_sum/(len(uav_validation_paths)//batch_size))

    print("Val acc:",acc_sum/(len(uav_validation_paths)//batch_size))

    print("Val loss:",val_loss_sum/(len(uav_validation_paths)//batch_size))

    if acc_sum/(len(uav_validation_paths)//batch_size) > max_val_acc:
        max_val_acc = acc_sum/(len(uav_validation_paths)//batch_size)
        print("Saved at epoch {}, learning rate = {}".format(epoch,learning_rate))
        torch.save(camera_model, "/Midgard/home/arohlen/camera_network_grid.pth.tar")
        #torch.save(camera_model, "/local_storage/users/arohlen/camera_network_grid.pth.tar")

    plt.plot(validation_acc,label="Validation accuracy")
    plt.plot(training_acc,label="Training accuracy")
    plt.legend()
    plt.title("Training and validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig("/Midgard/home/arohlen/validation_acc_grid.png")
    #plt.savefig(path+"validation_acc_grid.png")
    plt.close()
    if epoch % 4 == 3:
      learning_rate *= 0.7
      for g in optimizer.param_groups:
          g['lr'] = learning_rate

