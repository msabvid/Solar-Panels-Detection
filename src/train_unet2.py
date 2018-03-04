#TODO: change all paths to arguments
import sys
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
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
#from data_loader import ImagerLoader
from data_loader import *
import numpy as np
from dill.source import getsource
import tifffile as tiff


CLASSES = {1 : 'Solar Panel'}
os.chdir('/floydhub/CityFinancial/Solar-Panels-Detection/data/California')

output_path = 'images/label_cropped256/'  # labels path
data_path = 'images/images_cropped256/'   # input images path
dir_subimages = 'images/subimages/'

train_path = 'images/train_cropped256.csv'   # in each column: "image_name;image_label_name"
test_path = 'images/test_cropped256.csv'     # in each column: "image_name;image_label_name"




workers = 2
#epochs = 900
epochs=500
batch_size = 5 

#base_lr = 0.0015
base_lr = 0.001
momentum = 0.9
#weight_decay = 1e-4
weight_decay = 5e-4
print_freq = 10
prec1 = 0
best_prec1 = 0
lr = base_lr
count_test = 0
count = 0


my_transform = transforms.Compose([
    transforms.Scale(256)])

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
        y = y.view(batch_size,2,-1).contiguous()
        y = y.transpose(1,2).contiguous()
        y = y.view(-1,2).contiguous()
        return y

def main():
    global args, best_prec1
    best_F1 = 0
    

    model = Unet2()

    model.cuda()
    # optionally resume from a checkpoint
    #cudnn.benchmark = True

    train_loader = torch.utils.data.DataLoader(
        ImagerLoader(data_path,output_path,train_path, my_transform, crop=False,
                     normalize=True, size_cropped=256),
        batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True)#True)

    val_loader = torch.utils.data.DataLoader(
        ImagerLoader(data_path,output_path,test_path, my_transform, crop=False,
                     normalize=True, size_cropped=256),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)#True)

    
    # define loss function (criterion) and pptimizer
    weights = np.array([8.4,1.6])
    weights = np.array([9.0,1.0])
    weights
    weights = 1/weights
    print(weights)
    
    row_sums = weights.sum(axis=0)

    weights = weights/row_sums
    print('weights={}'.format(weights))
    weights = torch.Tensor(weights)

    criterion = nn.CrossEntropyLoss(weight=weights).cuda()

    #optimizer = torch.optim.SGD(model.parameters(), lr,
    #                            momentum=momentum,
    #                            weight_decay=weight_decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=weight_decay)
    #optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay,
    #                                momentum=momentum)
    for epoch in range(0, epochs):
        print('epoch={}'.format(epoch))

        #adjust_learning_rate(optimizer, epoch), divide it by 2 every 10 epochs
        # adjust learning rate
        lr = base_lr * (0.5 ** (epoch // 20))
        for param_group in optimizer.state_dict()['param_groups']:
            param_group['lr'] = lr

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)
    
        # remember best prec and save checkpoint to see when there starts to be overfitting every 5 epochs
        if epoch%5 == 0:
            F1 = validate(val_loader, model, criterion)
            is_best = F1>best_F1
            best_F1 = max(F1, best_F1)
            save_checkpoint({'epoch':epoch+1, 'state_dict':model.state_dict(), 'best_F1':best_F1}, is_best)                        


    predict(val_loader, model, criterion)
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
        # measure data loading time
        
        data_time.update(time.time() - end)
        input = input.cuda(async=True)
        target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)
        target_var = target_var.view(batch_size, -1)
        target_var = target_var.view(-1).contiguous()
        #target_var = target_var.view(batch_size, -1)
	#target_var = target_var.view(-1).contiguous()

        # compute output
        print('we compute output')
        output = model(input_var)
        print('first row of output {}'.format(output[0,:]))
        print('loss')
        loss = criterion(output, target_var)
        # measure accuracy and record loss
        prec1, recall1, F1 = accuracy(output.data, target, topk=(1, 1))

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
        print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))

        #return(model)




def predict(val_loader, model):
    checkpoint = torch.load('best_old_models/SGD/model_bes_long2.pth.tar')
    model.load_state_dict(checkpoint['state_dict']) 
    model.eval()
    sm = nn.Softmax()
    for i, (img_id, input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input = input.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        target_var = target_var.view(batch_size, -1)
        target_var = target_var.view(-1).contiguous()
        output = model(input_var)
        prec1, recall1, F1  = accuracy(output.data, target, topk=(1,1))
        _, pred = output.topk(1,1,True,True)
        probs = sm(output)
        height, width = (input.size()[2], input.size()[3]) 
        pred = pred.view(batch_size,1,height,width).data.cpu().numpy()
        probs = probs[:,1].contiguous().view(batch_size,1,height,width).data.cpu().numpy()
        for j in range(len(img_id)):
            pred_image = pred[j,:,:,:]
            probs_image = probs[j,:,:,:]
            name = img_id[j]
            tiff.imsave('images/prediction/pred_'+name+'.png', pred_image)     
            tiff.imsave('images/pix_probabilities/probs_'+name+'.png', probs_image)     


def predict_big_image(model, filename):
    val_loader = torch.utils.data.DataLoader(
        ImageLoaderPredictionBigImage(dir_subimages,filename, normalize=True),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True)#True) 
    checkpoint = torch.load('best_old_models/SGD/model_bes_long2.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    sm = nn.Softmax()
    for i, (img_id, input) in enumerate(val_loader):
        input = input.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        output = model(input_var)
        _, pred = output.topk(1,1,True, True)
        probs = sm(output)
        height, width = (input.size()[2], input.size()[3])
        pred = pred.view(batch_size, 1, height, width).data.cpu().numpy()
        probs = probs[:,1].contiguous().view(batch_size,1,height,width).data.cpu().numpy()
        for j in range(len(img_id)):
            pred_image = pred[j,:,:,:]
            probs_image = probs[j,:,:,:]
            name = img_id[j]
            tiff.imsave('images/prediction_big_images/'+name+'.png', pred_image)
            tiff.imsave('images/pix_probabilities_big_images/'+name+'.png', probs_image)
        
    reconstructed_prediction = reconstruct_image('images/prediction_big_images', filename)
    reconstructed_probabilities = reconstruct_image('images/pix_probabilities_big_images', filename)
    name_original_image = os.path.splitext(os.path.basename(filename))[0]
    tiff.imsave('images/prediction_big_images/'+name_original_image+'.png', reconstructed_prediction.astype('uint8'))
    tiff.imsave('images/pix_probabilities_big_images/'+name_original_image+'.png',reconstructed_probabilities)
    return 'sucess'
 

def validate(val_loader, model, criterion):
    global count_test
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    F_score = AverageMeter()
    #top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (img_id, input, target) in enumerate(val_loader):
        target = target.cuda(async=True)
        input = input.cuda(async=True)
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)
        target_var = target_var.view(-1).contiguous()
        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, recall1, F1  = accuracy(output.data, target, topk=(1,1))
        losses.update(loss.data[0], input.size(0))
        top1.update(prec1, input.size(0))
        F_score.update(F1, input.size(0))
        count_test = count_test+1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print('Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'.format(
                 i, len(val_loader), batch_time=batch_time, loss=losses,
                 top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    return F_score.avg




def save_checkpoint(state, is_best, filename='checkpoint_w.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best_long.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #global lr
    lr = base_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.state_dict()['param_groups']:
        param_group['lr'] = lr

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    batch_size = target.size(0)
    target = target.view(-1)
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
    print('TP: {}, FP: {}, P: {}, precision: {}, recall: {}, F1-score: {}, correct_p: {}'.format(TP, FP, P, precision, recall, F1, correct_p.sum()))
    return(precision, recall, F1)
    


def accuracy2(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    res = np.ones((10))*np.nan
    batch_size = target.size(0)
    print(output.float())
    max_output = np.argmax(output.float(),axis=1)
    target = target.int()
    for i,c in enumerate(CLASSES):
        assert target.shape== max_output.shape
        inter = np.logical_and(target==c,max_output==c).sum()
        union = np.logical_or(target==c,max_output==c).sum()
        res[i] = inter/union
    m = np.nanmean(res)
    return m


if __name__ == '__main__':
    main()
