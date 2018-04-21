#!/usr/local/bin/python
import numpy as np
import argparse
import cv2
import os
import errno
import random 

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def rotate(ps,img_size):
    deg = random.randint(0,360)
    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))
    ps = ps[0] - img_size/2
    return [(np.dot(ps,R.T)+img_size/2).astype(np.int)]

def triangle(sz,x=0,y=0):
    vrx = np.array(([sz/2+x,0+y],[sz+x,sz+y],[0+x,sz+y]))
    vrx = vrx.reshape((-1,1,2))
    return [vrx]

def rectangle(h,w,x=0,y=0):
    vrx = np.array(([0+x,0+y],[w+x,0+y],[w+x,h+y],[0+x,h+y]))
    vrx = vrx.reshape((-1,1,2))
    return [vrx]

if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--task", required=True, help="task [1]",type=int)
    ap.add_argument("-n", "--number", help="number of examples",type=int, default=100)
    ap.add_argument("-o", "--output", required=True, help="output folder",type=str)
    ap.add_argument("-z", "--size", help="image size",type=int,default=300)
    ap.add_argument("-s", "--shape-size", help="shape size",type=int,default=30)
    args = vars(ap.parse_args())

    mkdir_p(args['output'])
    if args['task']==1 or args['task']==2:
        x=0
        while x<args['number']:
            #make a triangle
	    img_t = np.zeros((args['size'],args['size'],1))
	    x_offset = random.randint(0,args['size']-args['shape_size'])
	    y_offset = random.randint(0,args['size']-args['shape_size'])
            t_ps = triangle(args['shape_size'],x_offset,y_offset)
            if args['task']==2:
                t_ps=rotate(t_ps,args['size'])
	    img_t = cv2.fillPoly(img_t, t_ps, (255,))

            #make a triangle
	    img_r = np.zeros((args['size'],args['size'],1))
	    x_offset = random.randint(0,args['size']-args['shape_size'])
	    y_offset = random.randint(0,args['size']-args['shape_size'])
            r_ps = rectangle(args['shape_size']/2,args['shape_size'],x_offset,y_offset)
            if args['task']==2:
                r_ps=rotate(r_ps,args['size'])
	    img_r = cv2.fillPoly(img_r, r_ps, (255,))

            img=np.maximum(img_r,img_t)
            if img.sum()<(img_t.sum()+img_r.sum()):
                print "SKIP"
                continue

	    cv2.imwrite('%s/%0.5d_input.png' % (args['output'],x),img)
	    cv2.imwrite('%s/%0.5d_triangle.png' % (args['output'],x),img_t)
	    cv2.imwrite('%s/%0.5d_rectangle.png' % (args['output'],x),img_r)
	    #cv2.imwrite('%s/%0.5d_out.png' % (args['output'],x),np.concatenate((img_r,img_t),axis=2))
            x+=1
	    #cv2.imshow('i',img)
	    #cv2.waitKey(5000)
