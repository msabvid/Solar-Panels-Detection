import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--nodes", action="store", type=int, default = 2, help="number of nodes")
parser.add_argument("--batch-size", action="store", type=int, default = 4, help="mini-batch size")
parser.add_argument("--log-interval", action="store", type=int, default=5)

args = parser.parse_args()

# we check that the number of nodes divides the mini batch size
assert(args.batch_size % args.nodes == 0)



batch_size = args.batch_size
nodes = args.nodes
lr = 0.01



comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    download=True, transform=transform) 

trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                                  shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                               download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                                 shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# get width and height of images
height, width = images.size()[2], images.size()[3]

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum = 0.9)




###################################
#if rank == 0:
##    dataiter = iter(trainloader)
##    images, labels = dataiter.next()
#    images_np = images.numpy()
#    labels_np = labels.numpy()
#    state_dict = net.state_dict(keep_vars=True)
#else:
#    images_np = None
#    labels_np = None
#    state_dict = None
#
#
#images_local = np.zeros([int(batch_size/nodes),3,height,width], dtype='float32')
#labels_local = np.zeros(int(batch_size/nodes), dtype='int64')
#
## scattering images and labels, and saving them in images_np, images_local
#comm.Scatter(images_np, images_local, root = 0)
#comm.Scatter(labels_np, labels_local, root = 0)
#state_dict = comm.bcast(state_dict, root=0)
#
#images_local = torch.from_numpy(images_local)
#labels_local = torch.from_numpy(labels_local)
#
## wrap them in a variable
#images_local, labels_local = Variable(images_local), Variable(labels_local)
#net.load_state_dict(state_dict)
#
## zero the parameter gradients
#net.zero_grad()
#
## forward + backward
#outputs= net(images_local)
#loss = criterion(outputs, labels_local)
#loss.backward()
#
#params = net.state_dict(keep_vars=True)
#gradients = dict()
#for key in params.keys():
#    gradients[key] = params[key].grad.data
#
##if rank==0:
##    print(params["conv1.bias"].grad)
##print('Success')
#
## gather the state dictionaries from all the processes in order to the SGD in the root process
#gradients_list = comm.gather(gradients, root=0)
#print(gradients_list[0]["conv1.bias"]) if rank==0 else ""
#with open(os.path.join(os.environ["HOME"],"process_"+str(rank)+".txt"),"w") as f:
#    f.write("process "+str(rank)+"\n")
#    f.write("======================\n")
#    if rank==0:
#        f.write("conv1.bias before update\n\n")
#        f.write(np.array_str(params["conv1.bias"].data.numpy()))
#        f.write("\n")
#    f.write("======================\n")
#    f.write("conv1.bias gradient\n\n")
#    f.write(np.array_str(params["conv1.bias"].grad.data.numpy()))
#    f.write("\n======================\n")
#
#
#if rank == 0:
#    #print(params_list[0]["conv1.bias"].grad)
#    #print(len(params_list[0]))
#    for gradient in gradients_list:
#        print(len(gradient))
#        for key in gradient.keys():
#            print(key)
#            #print(param[key].grad) if key=="conv1.bias" else "" 
#            #print(param[key].grad)
#            state_dict[key].data = state_dict[key].data - gradient[key]*0.01/len(gradients_list)
#    net.load_state_dict(state_dict)
#    with open(os.path.join(os.environ["HOME"],"process_"+str(rank)+".txt"),"a") as f:
#        f.write("conv1.bias after update \n\n")
#        f.write(np.array_str(state_dict["conv1.bias"].data.numpy()))
#        f.write("\n")
#
#
#####################################

for epoch in range(2):
    for i,(images,labels) in enumerate(trainloader):

        if rank == 0:
        #    dataiter = iter(trainloader)
        #    images, labels = dataiter.next()
            images_np = images.numpy()
            labels_np = labels.numpy()
            state_dict = net.state_dict(keep_vars=True)
        else:
            images_np = None
            labels_np = None
            state_dict = None


        images_local = np.zeros([int(batch_size/nodes),3,height,width], dtype='float32')
        labels_local = np.zeros(int(batch_size/nodes), dtype='int64')

        # scattering images and labels, and saving them in images_np, images_local
        comm.Scatter(images_np, images_local, root = 0)
        comm.Scatter(labels_np, labels_local, root = 0)
        state_dict = comm.bcast(state_dict, root=0)

        images_local = torch.from_numpy(images_local)
        labels_local = torch.from_numpy(labels_local)

        # wrap them in a variable
        images_local, labels_local = Variable(images_local), Variable(labels_local)
        net.load_state_dict(state_dict)

        # zero the parameter gradients
        net.zero_grad()

        # forward + backward
        outputs= net(images_local)
        loss = criterion(outputs, labels_local)
        loss.backward()

        params = net.state_dict(keep_vars=True)
        gradients = dict()
        for key in params.keys():
            gradients[key] = params[key].grad.data

        #if rank==0:
        #    print(params["conv1.bias"].grad)
        #print('Success')

        # gather the state dictionaries from all the processes in order to the SGD in the root process
        gradients_list = comm.gather(gradients, root=0)
        print(gradients_list[0]["conv1.bias"]) if rank==0 else ""
        with open(os.path.join(os.environ["HOME"],"process_"+str(rank)+".txt"),"a") as f:
            f.write("process "+str(rank)+"\n")
            f.write("======================\n")
            if rank==0:
                f.write("conv1.bias before update\n\n")
                f.write(np.array_str(params["conv1.bias"].data.numpy()))
                f.write("\n")
            f.write("======================\n")
            f.write("conv1.bias gradient\n\n")
            f.write(np.array_str(params["conv1.bias"].grad.data.numpy()))
            f.write("\n======================\n")


        if rank == 0:
            #print(params_list[0]["conv1.bias"].grad)
            #print(len(params_list[0]))
            for gradient in gradients_list:
                print(len(gradient))
                for key in gradient.keys():
                    print(key)
                    #print(param[key].grad) if key=="conv1.bias" else "" 
                    #print(param[key].grad)
                    state_dict[key].data = state_dict[key].data - gradient[key]*0.01/len(gradients_list)
            net.load_state_dict(state_dict)
            if i % args.log_interval == 0:
               images = Variable(images[:4,:,:,:])
               labels = Variable(labels[:4])
               outputs = net(images)
               loss = criterion(outputs, labels)
               with open(os.path.join(os.environ["HOME"], "cifar_log.txt"), "a") as f:
                   f.write('\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\n'.format(
                        epoch, i * len(images), len(trainloader.dataset),
                        100. * i / len(trainloader), loss.data[0]))
            with open(os.path.join(os.environ["HOME"],"process_"+str(rank)+".txt"),"a") as f:
                f.write("conv1.bias after update \n\n")
                f.write(np.array_str(state_dict["conv1.bias"].data.numpy()))
                f.write("\n")




print('success')

 

