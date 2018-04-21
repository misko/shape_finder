#!/usr/local/bin/python
import numpy as np
import argparse
import cv2
import os
import errno
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
        input = cv2.imread(data[k]['input'])[:,:,:1]
        triangle = cv2.imread(data[k]['triangle'])[:,:,:1]
        rectangle = cv2.imread(data[k]['rectangle'])[:,:,:1]
        return input,np.concatenate((triangle,rectangle),axis=2)

def n2t(im):
        return torch.from_numpy(np.transpose(im, (2, 0, 1)).astype(np.float)/255).float()

def t2n(im):
        n=im.cpu().data.numpy()
        n-=n.min()
        #n[n<0]=0
        div=255.0/max(1.0,n.max())
        return (np.transpose(n,(1,2,0))*div).astype(np.uint8)

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input-folder", help="input folder",type=str, default="./out")
    ap.add_argument("-mb", "--mini-batch", help="mb size",type=int, default=8)
    ap.add_argument("-s", "--save-dir", help="save dir",type=str,default="./out")
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
    optimizer = optim.SGD(model.parameters(), lr=args['learning_rate'], momentum=0.9)
    criterion=nn.MSELoss()
    for x in xrange(100000):
        optimizer.zero_grad()
        mini_loss=0
        for y in xrange(args['mini_batch']):
	    im_in,im_out=get_point()
	    im_in=Variable(n2t(im_in).unsqueeze(0))
	    im_out=Variable(n2t(im_out).unsqueeze(0))
	    output = model(im_in)
	    loss = criterion(output, im_out)
            mini_loss+=loss.data[0]/args['mini_batch']
            loss.backward()
        print mini_loss
        optimizer.step()
        im_in_np=t2n(im_in[0])
        im_out_np=t2n(im_out[0])
        im_pred_np=t2n(output[0])
        t=np.concatenate((im_in_np,im_out_np[:,:,:1],im_pred_np[:,:,:1]),axis=1)
        r=np.concatenate((im_in_np,im_out_np[:,:,1:2],im_pred_np[:,:,1:2]),axis=1)
        cv2.imshow('x',np.concatenate((t,r),axis=0))
        cv2.waitKey(10)
