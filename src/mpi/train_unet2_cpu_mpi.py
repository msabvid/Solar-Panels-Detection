#TODO: change all paths to arguments
import sys
#sys.path.append('/Users/msabate/Projects/CityFinancial/src')
sys.path.append('/floydhub/CityFinancial/Solar-Panels-Detection/src')

import argparse
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from data_loader import *
import numpy as np
import tifffile as tiff
from mpi4py import MPI
import argparse
import copy
import gc

parser = argparse.ArgumentParser()

parser.add_argument("--batch-size", action="store", type=int, default=5, help="mini-batch-size")
parser.add_argument("--log-interval", action="store", type=int, default=5)

args = parser.parse_args()
gc.enable()


# MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

assert(args.batch_size % size == 0)


CLASSES = {1 : 'Solar Panel'}
os.chdir('/lustre/home/ec002/msabate/Solar-Panels-Detection/data/California')

output_path = 'images/label_cropped256/'  # labels path
data_path = 'images/images_cropped256/'   # input images path
dir_subimages = 'images/subimages/'

train_path = 'images/train_cropped256.csv'   # in each column: "image_name;image_label_name"
test_path = 'images/test_cropped256.csv'     # in each column: "image_name;image_label_name"


workers = 2
#epochs = 900
epochs=120
batch_size = args.batch_size
batch_size_local = int(batch_size//size)

#base_lr = 0.0015
base_lr = 0.02
momentum = 0.9
gamma = 0.9
#weight_decay = 1e-4
weight_decay = 5e-4
print_freq = 10
prec1 = 0
best_prec1 = 0
lr = base_lr
count_test = 0
count = 0
torch.set_num_threads(10)

my_transform = transforms.Compose([
    transforms.Scale(256)])#,

class Unet2(nn.Module):

    def ConvLayer(self,nIn,nOut,dropout=False):
        net = nn.Sequential(
            nn.Conv2d(nIn,nOut,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(nOut),
            nn.ReLU(inplace=True),
            nn.Conv2d(nOut,nOut,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(nOut),
            nn.ReLU(inplace=True))
        return net


    def __init__(self, num_classes=1):
        super(Unet2, self).__init__()

        #self.D1 = self.ConvLayer(20,32,dropout=False)
        self.D1 = self.ConvLayer(3,32,dropout=False)
        self.D2 = self.ConvLayer(32,64,dropout=False)
        self.D3 = self.ConvLayer(64,128,dropout=False)
        self.D4 = self.ConvLayer(128,256,dropout=False)
        self.B = self.ConvLayer(256,512,dropout=False)

        self.U4_input = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
            )

        self.U3_input = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
            )
        self.U2_input = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
            )
        self.U1_input = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
            )

        self.U1 = self.ConvLayer(64,32,dropout=False)
        self.U2 = self.ConvLayer(128,64,dropout=False)
        self.U3 = self.ConvLayer(256,128,dropout=False)
        self.U4 = self.ConvLayer(512,256,dropout=False)
        self.last = nn.Conv2d(32,2,kernel_size=1)

    def forward(self, x):
        d1 = self.D1(x)
        d1_pool = nn.MaxPool2d(kernel_size=2, stride=2)(d1)
        d2 = self.D2(d1_pool)
        d2_pool = nn.MaxPool2d(kernel_size=2, stride=2)(d2)
        d3 = self.D3(d2_pool)
        d3_pool = nn.MaxPool2d(kernel_size=2, stride=2)(d3)
        d4 = self.D4(d3_pool)
        d4_pool = nn.MaxPool2d(kernel_size=2, stride=2)(d4)
        b = self.B(d4_pool)
        u4_input = self.U4_input(b)
        u4_input = u4_input.contiguous()
        u4 = self.U4(torch.cat((d4,u4_input),1))
        u3_input = self.U3_input(u4)
        u3 = self.U3(torch.cat((d3,u3_input),1))
        u3 = u3.contiguous()
        u2_input = self.U2_input(u3)
        u2 = self.U2(torch.cat((d2,u2_input),1))
        u1_input = self.U1_input(u2)
        u1_input = u1_input.contiguous()
        u1 = self.U1(torch.cat((d1,u1_input),1))
        y = self.last(u1)
        y = y.view(batch_size_local,2,-1).contiguous()
        y = y.transpose(1,2).contiguous()
        y = y.view(-1,2).contiguous()
        return y





def main():
    global args, best_prec1
    best_F1 = 0

    model = Unet2()


    train_loader = torch.utils.data.DataLoader(
        ImagerLoader(data_path,output_path,train_path, my_transform, crop=False,
                     normalize=True, size_cropped=256),
        batch_size=batch_size, shuffle=True,
        num_workers=workers)#, pin_memory=True)#True)

    val_loader = torch.utils.data.DataLoader(
        ImagerLoader(data_path,output_path,test_path, my_transform, crop=False,
                     normalize=True, size_cropped=256),
        batch_size=batch_size, shuffle=False,
        num_workers=workers)#, pin_memory=True)#True)

    # define loss function (criterion) and optimizer
    weights = np.array([9.,1.])
    weights = 1/weights
    #print(weights)
    row_sums = weights.sum(axis=0)
    weights = weights/row_sums
    print('weights={}'.format(weights))
    weights = torch.Tensor(weights)

    criterion = nn.CrossEntropyLoss(weight=weights)#.cuda()

    #optimizer = torch.optim.SGD(model.parameters(), lr=0.001,
    #                            momentum=momentum,
    #                            weight_decay=weight_decay)

    # height and width of images
    _, input, target = iter(train_loader).next()
    height,width = input.size()[2], input.size()[3]
    first = True
    model.train()
    for epoch in range(0, epochs):
        epoch_start_time = time.time()
        if rank == 0:
            with open(os.path.join(os.getcwd(), "..", "working", "log_size_"+str(size)+"_global_epoch.txt"), "a") as f:
                f.write('Starting Epoch: [{0}]\n'.format(epoch))
        lr = base_lr * (0.5 ** (epoch // 20))
        print('epoch={}'.format(epoch))
        #print('number of threads: {}'.format(torch.get_num_threads()))
        for i, (img_id, input, target) in enumerate(train_loader):
            if rank == 0:
                with open(os.path.join(os.getcwd(), "..", "working", "log_size_"+str(size)+"_global_epoch.txt"), "a") as f:
                    f.write('Epoch: {} \tNew minibatch {}\n'.format(epoch, i))
            end=time.time()
            if rank == 0:
                inputs_np = input.numpy()
                targets_np = target.numpy()
                state_dict = copy.deepcopy(model.state_dict(keep_vars=True))
            else:
                inputs_np = None
                targets_np = None
                state_dict = None

            inputs_local = np.zeros([int(batch_size/size), 3, height, width], dtype='float32')
            targets_local = np.zeros([int(batch_size/size),height, width], dtype='int64')

            # scattering images and labels and saving them in inputs_local, labels_local
            comm.Scatter(inputs_np, inputs_local, root=0)
            comm.Scatter(targets_np, targets_local, root=0)
            state_dict = comm.bcast(state_dict, root=0)
            if rank == 0:
                with open(os.path.join(os.getcwd(), "..", "working", "log_size_"+str(size)+"_global_epoch.txt"), "a") as f:
                    f.write('Epoch: {} \tMinibatch {}\tMinibatch and model state dict scattered and broadcasted\n'.format(epoch, i))

            inputs_local = torch.from_numpy(inputs_local)
            targets_local = torch.from_numpy(targets_local)

            # wrap them in a variable
            inputs_local, targets_local = torch.autograd.Variable(inputs_local), torch.autograd.Variable(targets_local)
            targets_local = targets_local.view(int(batch_size/size), -1)
            targets_local = targets_local.view(-1)

            # we load the model dictionary to load the weights updated in the root process
            model.load_state_dict(state_dict)

            # zero the parameter gradients
            model.zero_grad()

            # forward + backward
            #print("size of inputs_local: {}".format(inputs_local.size()))
            init_forward = time.time()
            outputs = model(inputs_local)
            time_forward = time.time() - init_forward
            print("success forward")
            #print("size of outputs: {}".format(outputs.size()))
            #print("size of targets_local: {}".format(targets_local.size()))
            loss = criterion(outputs, targets_local)
            print("success loss")
            init_back = time.time()
            loss.backward()
            time_back = time.time() - init_back
            print("sucess backwards")

            # we get the gradients from the backpropagation
            params = model.state_dict(keep_vars=True)
            gradients = dict()
            for key in params.keys():
                #print("key = {}".format(key))
                try:
                    gradients[key] = params[key].grad.data
                except AttributeError: # we have the batchNorm layer that is not a Variable (and hence does not hava gradient), as it only does a normalization
                    pass

            # gather the gradients from all the processes in order to do the SGD step in the root process (rank=0)
            init_gather = time.time()
            gradients_list = comm.gather(gradients, root=0)
            time_gather = time.time() - init_gather
            if rank == 0:
#                # plain SGD
#                for gradient in gradients_list:
#                    for key in gradient.keys():
#                        state_dict[key].data = state_dict[key].data - gradient[key]*lr/len(gradients_list)
                # SGD with momentum
                # 1. we update v (speed): v := \gamma v - \alpha gradient(\theta)
                if first:
                    v = copy.deepcopy(state_dict)
                    for key in v.keys():
                        if not torch.is_tensor(v[key]):
                            v[key].data = v[key].data*0
                            for gradient in gradients_list:
                                v[key].data = v[key].data+gradient[key]*lr
                            v[key].data = v[key].data/len(gradients_list)
                    first = False
                else:
                    for key in v.keys():
                        if not torch.is_tensor(v[key]):
                            v[key].data = v[key].data*gamma
                    for key in v.keys():
                        if not torch.is_tensor(v[key]):
                            for gradient in gradients_list:
                                v[key].data = v[key].data+gradient[key]*lr/len(gradients_list)
                # 2. we update the parameters using the speed (v): \theta := \theta - v
                for key in state_dict.keys():
                    if not torch.is_tensor(v[key]):
                        state_dict[key].data = state_dict[key].data - v[key].data



                model.load_state_dict(state_dict)
#                if i % args.log_interval == 0:
#                    time_batches = time.time() - end
#                    inputs_subset = Variable(input[:int(batch_size/size),:,:,:])
#                    target_subset = Variable(target[:int(batch_size/size)]) 
#                    outputs = model(inputs_subset)
#                    loss = criterion(outputs, target_subset)
#                    # we print accuracy measures on the first elments of the mini-batch
#                    prec1, recall1, F1 = accuracy(outputs.data, target_subset, topk=(1, 1))

            else:
                v = None
            batch_time = time.time() - end
            prec1, recall1, F1 = accuracy(outputs.data, targets_local.data, topk=(1,1))
            with open(os.path.join(os.getcwd(), "..", "working", "log_size_"+str(size)+"_rank_"+str(rank)+".txt"), "a") as f:
                f.write('Epoch: [{0}][{1}/{2}]\t'
                  'Time_all {batch_time:.3f}\t'
                  'Time forward pass {time_forward:.3f}\t'
                  'Time backward pass {time_back:.3f}\t'
                  'Time gather gradients {time_gather:.3f}\t'
                  'Loss {loss:.4f}\t'
                  'Prec@1 {prec1:.3f}\t'
                  'Recall1 {recall1:.3f}\t'
                  'F1 {F1:.3f}\n'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   time_forward=time_forward,
                   time_back=time_back, time_gather=time_gather,
                   loss=loss.data[0], prec1=prec1, recall1=recall1, F1=F1))
        epoch_time = time.time() - epoch_start_time
        if rank == 0:
            with open(os.path.join(os.getcwd(), "..", "working", "log_size_"+str(size)+"_global_epoch.txt"), "a") as f:
                f.write('Ending Epoch: [{0}]\t Time epoch: {time_epoch:.3f}\n'.format(epoch, time_epoch = epoch_time))
        gc.collect()


        #adjust_learning_rate(optimizer, epoch), divide it by 2 every 20 epochs
        # adjust learning rate


    #predict(val_loader, model, criterion)
    print('training finished')

def train(train_loader, model, criterion, optimizer, epoch):
    global count
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    F_score = AverageMeter()
    #top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (img_id, input, target) in enumerate(train_loader):
        print('i={}'.format(i))
        print('num of threads = {}'.format(torch.get_num_threads()))
        # measure data loading time

        data_time.update(time.time() - end)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_var = target_var.view(batch_size, -1)
        target_var = target_var.view(-1)#.contiguous()

        # compute output
        print('we compute output')
        output = model(input_var)
        print('first row of output {}'.format(output[0,:]))
        print('loss')
        loss = criterion(output, target_var)
        prec1, recall1, F1 = accuracy(output.data, target.data, topk=(1, 1))

        losses.update(loss.data[0], input.size(0))
        top1.update(prec1, input.size(0))   # input.size(0) = nbatches
        F_score.update(F1, input.size(0))   # input.size(0) = nbatches
        # compute gradient and do SGD step
        optimizer.zero_grad()
        print('backward propagation')
        loss.backward()
        print('parameters update')
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        count = count+1
        with ('/lustre/home/ec002/msabate/Solar-Panels-Detection/Hogwild_output.txt', 'a') as f:
            f.write('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                       epoch, i, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, top1=top1))

        #return(model)





def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    target = target.view(-1)
    #output = output.view(batch_size,2,-1).contiguous()
    #output = output.transpose(1,2).contiguous()
    #output = output.view(-1,2).contiguous()

    target = target.view(-1).contiguous()

    _, pred = output.topk(1, 1, True, True)
    #pred = pred.t()
    #print(pred)
    suma = 0
    count = 0
    #res = np.ones((10))*np.nan


    #print(res)
    #return suma/count
    c = 1  # first class of CLASSES
    correct_p = pred.eq(c)
    correct_t = target.eq(c)
    TP = correct_p.add(correct_t.view(-1,1)).eq(2).view(-1).sum()
    FP = correct_p.sum()-TP
    P = correct_t.sum()
    FN = P - TP
    try:
        precision= TP/(TP+FP)
    except:
        precision = np.nan
    try:
        recall = TP/(TP+FN)
    except:
        recall = np.nan
    try:
        F1 = 2/(1/precision + 1/recall)
    except:
        F1 = np.nan

    return(precision, recall, F1)



if __name__ == '__main__':
    print('hola')
    main()
