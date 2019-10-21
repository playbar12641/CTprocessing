# -*- coding: utf-8 -*-
"""
Created on Sat Oct 19 22:12:29 2019

@author: Xiao Luo Oct 2019

"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage

def read_pile(path,file_name,filter=None):

    #file_path = filedialog.askopenfilename()
    file_path = path+'\\'+file_name
    #print(file_path)

    file_size = os.path.getsize(file_path)
    total_num_slice = int(math.floor(file_size/(1024*(512+10)))+1)  #512 by 512
    #print(total_num_slice)
    in_file = open(file_path, "rb") # opening for [r]eading as [b]inary
    dt = np.dtype('<i2')
    data_1 = np.fromfile(file_path, dtype=dt, count=512*512*total_num_slice)
    # data_2 = data_1.astype('i2')
    data_double=np.reshape(data_1, (512,512,total_num_slice),order='F') # the CT number this data is unfiltered
    
    if filter!=None:
        for i in range(total_num_slice):
            data_double[:,:,i] = ndimage.median_filter(data_double[:,:,i],filter)
    
    #print(data_double.shape)
    #print(data_double[0, 0, 0])
    #m1=data_double[:, :, 0] #only get first slice

    #np.save(file_name,data_double) 

    return data_double

def print_stats(scan,n):
    nr, nc, nz = scan.shape
    temp = scan[:,:,n]    
    temp = temp.reshape(nr*nc)
    temp = pd.DataFrame(temp)
    
    return print(temp.describe())

def remove_outlier_std(scan,std_mplier,t_set):
    # remove outlier based on std and mean assuming a Gassian distribution
    nr, nc, nz = scan.shape    
    scan2 = 0*scan    
    for i in range(nz):
        temp = scan[:,:,i]
        std = np.std(temp)
        mean = np.std(temp)
        up_lim = mean+std_mplier*std
        lo_lim = mean-std_mplier*std    
        temp[temp>up_lim] = t_set
        temp[temp<lo_lim] = t_set    
        scan2[:,:,i] = temp   
    return scan2

def remove_outlier_percentile(scan,lowp,highp,t_set):
    nr, nc, nz = scan.shape    
    scan2 = 0*scan    
    for i in range(nz):
        temp = scan[:,:,i]
        up_lim = np.percentile(temp,highp)
        lo_lim = np.percentile(temp,lowp) 
        temp[temp>up_lim] = t_set
        temp[temp<lo_lim] = t_set    
        scan2[:,:,i] = temp   
    return scan2    
    
def reindex_scan(scan):
    # turn a scan 512 by 512 by nslice to 2d matrix, with first two column of x and y indices
    # and column 3 so forth to be the greyscale value
    
    nr, nc, nz = scan.shape 
    col_names = ['slice' + x for x in (np.arange(nz)+1).astype(str)]
    col_names = ['x','y']+col_names
    reindexed = np.zeros((nr*nc,nz+2))
    row, col = np.indices((nr, nc))
    reindexed[:,0] = row.reshape(nr*nc).astype(int)
    reindexed[:,1] = col.reshape(nr*nc).astype(int)    
    for i in range(nz):    
        reindexed[:,i+2] = scan[:,:,i].reshape(nr*nc)
    
    reindexed = pd.DataFrame(data=reindexed,columns=col_names)
    return reindexed


def create_circular_mask(h, w, center=None, radius=None):

    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    #mask[mask==0]=np.nan
    return mask

def create_square_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = [int(w/2), int(h/2)]
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    
    
    center_y = center[1];
    center_x = center[0];
    
    mask = np.full((int(h), int(w)), False, dtype=bool)
    radius = int(radius)
    mask[center_y-radius:center_y+radius+1, center_x-radius:center_x+radius+1] = True
    return mask


def create_mask_matrix(nr, nc, nz, circle_centers, r, t):
    mask_mat = np.full((nr,nc,nz),False,dtype=bool)
    
    for i in range(nz):
        center = [circle_centers[i,0], circle_centers[i,1]]
        
        if t=='square':
            mask_temp = create_square_mask(nr, nc, center, radius=r)
        elif t=='circle':
            mask_temp = create_circular_mask(nr, nc, center, radius=r)
        else:
            print('input error, please verify mask shape')
        
        mask_mat[:,:,i] = mask_temp        
        
    return mask_mat


def pick_circle(xi,yi,dx,dy,nx,ny,r,dn):
    # you have to have a good guess on the initial circle center and a radius     
    nr,nc = dn.shape
    x_centers = xi+np.arange(-1*dx*nx,dx*nx+1 ,dx)
    y_centers = yi+np.arange(-1*dy*ny,dy*ny+1 ,dy)

    var_mat = np.zeros((len(x_centers),len(y_centers)))
    for i in range(len(x_centers)):
        for j in range(len(y_centers)):
            center_temp=[x_centers[i],y_centers[j]];
            mask_temp = create_circular_mask(nr, nc, center=center_temp, radius=r)
            var_mat[i,j]=np.var(dn[mask_temp])        

    ind = np.unravel_index(np.argmin(var_mat, axis=None), var_mat.shape)
    ix = ind[0];iy = ind[1]
    xcf = x_centers[ix]
    ycf = y_centers[iy]
    return xcf,ycf


def crop_scan(scan,circular_mask,square_mask):
    scan[~circular_mask] = float('nan')
    rs = np.sqrt(np.sum(square_mask[:,:,0])).astype(int)
    nr, nc, nz = scan.shape
    croped = np.zeros((rs,rs,nz))
    
    for i in range(nz):
        sqm = square_mask[:,:,i]
        temp = scan[:,:,i]        
        croped[:,:,i] = np.reshape(temp[sqm],((rs,rs)))
    
    return croped
