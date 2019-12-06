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
import csv
from datetime import datetime

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

# some old codes are commented here
# this also plot pressure dat file
# import tkinter as tk
# from tkinter import filedialog
# import csv
# from datetime import datetime
# import matplotlib.pyplot as plt

# def read_dat_file(t_col,p_col,l_start,dn,mutip):
#     root = tk.Tk()
#     root.withdraw()
#     file_path = filedialog.askopenfilename()
#     with open(file_path) as f:
#         reader = csv.reader(f, delimiter="\t")
#         for row in reader:
#              columns = list(zip(*reader))
     

#     line_start=l_start
#     time=columns[t_col][line_start:-1]
#     delP=columns[p_col][line_start:-1]
#     temp=np.array(delP)
#     delP=np.asfarray(temp,float)

#     n=len(time)

#     time=time[0:n:dn]
#     delP=delP[0:n:dn]
    
#     delP = [x * mutip for x in delP]
#     t_minute=[0]*len(time)
    
    
#     fmt = '%m/%d/%Y %H:%M:%S.%f'
#     d1 = datetime.strptime(time[0], fmt)

#     for i in range(len(time)):
#         d2 = datetime.strptime(time[i], fmt)
#         t_minute[i]=(d2-d1).seconds/60+(d2-d1).days*24*60

#     return np.array(t_minute), np.array(delP)

def read_pressure_dat_file(fn,t_col,p_col,l_start,dn,mutip):
#t_inv,p_inv=read_pressure_dat_file(fn,t_col=1,p_col=3,l_start=4,dn=1,mutip=1)
    with open(fn) as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
             columns = list(zip(*reader))
     
    line_start=l_start
    time=columns[t_col][line_start:-1]
    delP=columns[p_col][line_start:-1]
    temp=np.array(delP)
    delP=np.asfarray(temp,float)

    n=len(time)

    time=time[0:n:dn]
    delP=delP[0:n:dn]
    
    delP = [x * mutip for x in delP]
    t_minute=[0]*len(time)
    
    
    fmt = '%m/%d/%Y %H:%M:%S.%f'
    d1 = datetime.strptime(time[0], fmt)

    for i in range(len(time)):
        d2 = datetime.strptime(time[i], fmt)
        t_minute[i]=(d2-d1).seconds/60+(d2-d1).days*24*60
   
    return np.array(t_minute), np.array(delP)


def make_profile_plot(y,df,fig_prop):
    lxc=['b','g','r','c','m','k']
    lxls=['-','--','-.',':']
    lxmarker=['.','s','x','^','d','P']
    
    fig=plt.figure()
    ax=fig.add_subplot(111)
      
    for n in np.arange(len(df.columns)):
        n_color=np.remainder(n,len(lxc)).astype(int)
        n_ls=np.remainder(n,len(lxls)).astype(int)
        n_marker=np.remainder(n,len(lxmarker)).astype(int)
        ax.plot(y,df.iloc[:,n],c=lxc[n_color],ls=lxls[n_ls],marker=lxmarker[n_marker],label=df.columns[n],markersize=fig_prop.markersize)
    
    
    
    ax.set_ylim(fig_prop.ylim)
    ax.set_xlim(fig_prop.xlim)

    ax.set_xlabel(fig_prop.xlabel)
    ax.set_ylabel(fig_prop.ylabel)
    
    plt.legend(loc=fig_prop.legend_loc)
    plt.title(fig_prop.title)
    plt.subplots_adjust(left=0.0, bottom=0.0, right=0.8, top=1, wspace=0.2, hspace=0.2)

#     plt.draw()
#     plt.show()

    
    
    
    
    
    
    
