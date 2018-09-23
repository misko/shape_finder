#!/usr/local/bin/python
import numpy as np
import argparse
import cv2
import os
import errno
import shutil
import random 
import torch
from torch.autograd import Variable
from mnet import MNET
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

data={}

def get_point():
    while True:
        k=random.choice(data.keys())
        if 'input' not in data[k] or 'rectangle' not in data[k] or 'triangle' not in data[k]:
            continue
        input = cv2.imread(data[k]['input'])[None,:,:,:1]
        triangle = cv2.imread(data[k]['triangle'])[None,:,:,:1]
        rectangle = cv2.imread(data[k]['rectangle'])[None,:,:,:1]
        return input,np.concatenate((triangle,rectangle),axis=3)
        #return input,np.concatenate((triangle,triangle),axis=3)

def n2t(im):
        #return torch.from_numpy(np.transpose(im, (2, 0, 1)).astype(np.float)/255).float()
	if args['cuda']==1:
        	return torch.from_numpy(np.transpose(im, (0,3, 1, 2)).astype(np.float)/255).float().cuda()
	else:
        	return torch.from_numpy(np.transpose(im, (0,3, 1, 2)).astype(np.float)/255).float()
        #return torch.from_numpy(np.transpose(im, (0,3, 1, 2)).astype(np.float)).float()

def t2n(im):
        n=im.cpu().data.numpy()
        n-=n.min()
        #n[n<0]=0
        div=255.0/max(1.0,n.max())
        #return (np.transpose(n,(1,2,0))*div).astype(np.uint8)
        return (np.transpose(n,(0,2,3,1))*div).astype(np.uint8)

#https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3
# <3 mratsimMamy Ratsimbazafy
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')



if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input-folder", help="input folder",type=str, default="./out")
    ap.add_argument("-mb", "--mini-batch", help="mb size",type=int, default=8)
    ap.add_argument("-s", "--save-dir", help="save dir",type=str,default="./out")
    ap.add_argument("-c", "--cuda", help="cuda?",type=int,default=0)
    ap.add_argument("-x", "--show", help="show?",type=int,default=0)
    ap.add_argument("-r", "--resume", help="resume",type=str,default="")
    ap.add_argument("-lr", "--learning-rate", help="save dir",type=float,default=0.001)
    args = vars(ap.parse_args())

    mkdir_p(args['save_dir'])
    # traverse root directory, and list directories as dirs and files as files
    for root, dirs, files in os.walk(args['input_folder']):
	path = root.split(os.sep)
	for file in files:
            try:
                idx,ty=file.split('.')[-2].split('_')
                if idx not in data:
                    data[idx]={}
                data[idx][ty]=root+'/'+file
            except:
                pass

    model=MNET()
    if args['cuda']==1:
       model= model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args['learning_rate'], momentum=0.9)
    if args['resume']!='':
        if os.path.isfile(args['resume']):
            print("=> loading checkpoint '{}'".format(args['resume']))
            checkpoint = None
            if args['cuda']==0:
                checkpoint = torch.load(args['resume'], map_location=lambda storage, loc: storage)
            else:
                checkpoint = torch.load(args['resume'],  map_location=lambda storage, location: 'gpu')
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args['resume'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args['resume']))
    #optimizer = torch.optim.Adam(model.parameters(), lr=args['learning_rate'])
    criterion=nn.MSELoss()
    #criterion=nn.BCELoss()
    #criterion=nn.BCEWithLogitsLoss()
    for x in xrange(100000):
        im_ins=[]
        im_outs=[]
        for y in xrange(args['mini_batch']):
	    im_in,im_out=get_point()
            im_ins.append(im_in)
            im_outs.append(im_out)
        im_in=Variable(n2t(np.concatenate(im_ins,axis=0)))
        im_out=Variable(n2t(np.concatenate(im_outs,axis=0)))
        print im_in.size(),im_out.size()

	output = model(im_in)
	loss = criterion(output, im_out)
        #loss = F.cross_entropy(output, im_out) #, weight=weight, size_average=self.size_average)
        #loss = F.nll_loss(output, im_out) #, weight=weight, size_average=self.size_average)
        mini_loss = loss.data[0] #/args['mini_batch']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        im_in_np=t2n(im_in)
        im_out_np=t2n(im_out)
        im_pred_np=t2n(output) #.astype(np.float32)
        print mini_loss,im_pred_np.mean(),im_pred_np.max()
        #im_pred_np=(((im_pred_np-im_pred_np.min())/im_pred_np.max())*255).astype(np.uint8)
        t=np.concatenate((im_in_np[0],im_out_np[0,:,:,:1],im_pred_np[0,:,:,:1]),axis=1)
        r=np.concatenate((im_in_np[0],im_out_np[0,:,:,1:2],im_pred_np[0,:,:,1:2]),axis=1)
        if args['show']==1:
            cv2.imshow('x',np.concatenate((t,r),axis=0))
            cv2.waitKey(10)
	if x%100:
	    save_checkpoint({
            'epoch': x + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
            }, True)
