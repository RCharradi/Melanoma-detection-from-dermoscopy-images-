# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 18:17:06 2019

@author: Ramzi Charradi
"""
import numpy as np
from skimage.transform import resize
import cv2
import shutil


######################################################################################
######################################################################################
###########        *** Copy images to directories ***               ##################
######################################################################################
######################################################################################

# For deep learning we need separate folder for train,validation and submission images
#each folder contains two folders one called benign and the other called melanoma

def copy_valid(y,X,src,dest):
    for i in range(600, X.shape[0]):
        image = X[i]
        if (y[i]==0):
            destination = dest + '/benign'
        else:
            destination = dest + '/malignant'
        file = src + str(image) + '.jpg'
        shutil.copy(file,destination)
        
def copy_train(y,X,src,dest):
    for i in range(0, 600):
        image = X[i]
        if (y[i]==0):
            destination = dest + '/benign'
        else:
            destination = dest + '/malignant'
        file = src + str(image) + '.jpg'
        shutil.copy(file,destination)
        
def copy_submission(y,X,src,dest):
    for i in range(0, X.shape[0]):
        image = X[i]
        if (y[i]==0):
            destination = dest + '/benign'
        else:
            destination = dest + '/malignant'
        file = src + str(image) + '.jpg'
        shutil.copy(file,destination)
        

######################################################################################
######################################################################################
###########        ***  Build matrix from images  ***               ##################
######################################################################################
######################################################################################
        
        
def build_matrix(X):
    # load images
    M = []
    for i in range(0, X.shape[0]):
        image = X[i]
        im = cv2.imread('./images/im/'  + str(image) + '.jpg')
        img_down = resize(im,(224,224), mode='reflect')
        M.append(img_down)
    return np.asarray(M)
  
def build_matrix_mask(X):
    # load images
    
    M = []
    for i in range(0, X.shape[0]):
        image = X[i]
        im = cv2.imread('./images/im/'  + str(image) + '_segmentation.jpg',0)
        img_down = resize(im,(224,224), mode='reflect')
        M.append(img_down)
    return np.asarray(M)