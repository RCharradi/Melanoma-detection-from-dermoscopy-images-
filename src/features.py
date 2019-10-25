# -*- coding: utf-8 -*-
"""
Created on Fri MAY 3 22:15:33 2019

@author: Ramzi Charradi
"""

import numpy as np
from scipy import ndimage
from skimage.io import imread
from skimage.measure import shannon_entropy
from skimage.measure import regionprops
from scipy.stats import gaussian_kde
import cv2

######################################################################################
######################################################################################
###########                      ***Assimetry ***                   ##################
######################################################################################
######################################################################################

############################# Assimetry of shape #####################################

def compute_deltaS1(img_gray, mask):
    
    (n,m) = img_gray.shape
    
    ## find contours and sort them
    cnts, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours = sorted(cnts, key=cv2.contourArea, reverse=True)
    cnt = contours[0][:,0]
    area = cv2.contourArea(cnt)
    
    # extracts moments of the mask
    M = cv2.moments(mask)
    
    # coordinates of moments
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    
    deltaS1 = 0
    for i in range(0,min(cx,m-cx)):
        for j in range(0,min(cy,n-cy)):
            if ((mask[cy+j,cx+i] != 0) or (mask[cy+j,cx+i] != 0)):
                deltaS1 += abs(img_gray[cy+j,cx+i]-img_gray[cy-j,cx+i])
                
    #Normalize
    deltaS1 = deltaS1/area
    
    return deltaS1

def compute_delta_S1S2(img_gray, mask):
    
    deltalist= [] 
    
    for ang in range(18):
        img_gray_rot = ndimage.rotate(img_gray, ang*10) 
        img_mask_rot = ndimage.rotate(mask, ang*10)
        DS = compute_deltaS1(img_gray_rot,img_mask_rot)
        deltalist.append(DS) 

    DS1 = deltalist[0]
    DS2 = deltalist[9]
    min = DS1+DS2 #
    
    for i in range(1,len(deltalist)):
        # for each couple DeltaS1 and DeltaS2, try to find a smaller average value
        if (deltalist[i]+deltalist[(i+9)%len(deltalist)]) < min :
            DS1 = deltalist[i]
            DS2 = deltalist[(i+9)%len(deltalist)]
            min = DS1+DS2
    if (DS1>DS2):
        return (DS2,DS1)
    else:
        return (DS1,DS2)

def assimitry_Shape(im_name) :
    

    print('__Assimetry of shape__ :',im_name)
    
    # upload image and convert it to grey level
    filename = './images/im/{}.jpg'.format(im_name)
    m_colored = imread(filename)
    m_gray = cv2.cvtColor(m_colored,cv2.COLOR_RGB2GRAY)
    
    # upload masj
    filemask = './images/im/{}_Segmentation.jpg'.format(im_name)
    cvu8_mask = imread(filemask).astype(np.uint8)
    thresh, masku8 = cv2.threshold(cvu8_mask, 127, 255, cv2.THRESH_BINARY)
    
    return compute_delta_S1S2(m_gray,masku8)

######################### Asimmetry of color intensity,#############################
    
def compute_deltaC1(img_grayscale, binarymask):

    M = cv2.moments(binarymask)
    c = int(M['m01']/M['m00'])
    
    #take half the top image, half the bottom image
    A12 = img_grayscale[:c,:]
    A34 = img_grayscale[c:,:]
    A12masked = A12[(binarymask[:c,:]).astype(bool)]
    A34masked = A34[(binarymask[c:,:]).astype(bool)]
    A12masked.sort()
    A34masked.sort()
    # use Gaussian Kernel Density estimation
    kde = gaussian_kde(A12masked)
    kde2 = gaussian_kde(A34masked)
    
    res = np.sum(abs(kde.evaluate(np.arange(256))-kde2.evaluate(np.arange(256))))
    return res


def delta_C1_C2(img_gray, mask):
    x,y,w,h = cv2.boundingRect(mask)
    newmask = mask[y:y+h,x:x+w]
    newimg = img_gray[y:y+h,x:x+w]
    deltalist= [] #list of the delta 1 and 2 (only calculated once)
    for ang in range(18):
        img_gray_rot = ndimage.rotate(newimg, ang*10) #each step rotates by 10° mask and img
        img_mask_rot = ndimage.rotate(newmask, ang*10)
        DC = compute_deltaC1(img_gray_rot,img_mask_rot)
        deltalist.append(DC) #add the calculated delta
    DC1 = deltalist[0]
    DC2 = deltalist[9]
    min = DC1+DC2 # init min (as the average between 1 and 2)
    for i in range(1,len(deltalist)):
        if (deltalist[i]+deltalist[(i+9)%len(deltalist)]) < min :
            DC1 = deltalist[i]
            DC2 = deltalist[(i+9)%len(deltalist)]
            min = DC1+DC2
    if (DC1>DC2):
        return (DC2,DC1)
    else:
        return (DC1,DC2)

def assimetry_color(im_name) :
    
    print('__Assimetry of color __ :',im_name)
    
    # upload image and convert it to grey level
    filename = './images/im/{}.jpg'.format(im_name)
    m_colored = imread(filename)
    m_gray = cv2.cvtColor(m_colored,cv2.COLOR_RGB2GRAY)
    
    # upload masj
    filemask = './images/im/{}_Segmentation.jpg'.format(im_name)
    cvu8_mask = imread(filemask).astype(np.uint8)
    #since all the mask pixel values are not either 0 or 255, a threshold is needed to get a binary mask
    thresh, masku8 = cv2.threshold(cvu8_mask, 127, 255, cv2.THRESH_BINARY)
    
    return delta_C1_C2(m_gray,masku8)

######################################################################################
######################################################################################
###########            ***Colors, peripheral versus central***            ############
######################################################################################
######################################################################################
    
##################       features f13, f14 and f15     ##############################♣
    
def erode_mask(mask):
    
    kernel = np.ones((5,5),np.uint8)
    
    cnts, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    contours = sorted(cnts, key=cv2.contourArea, reverse=True)
    cnt = contours[0][:,0]
    full_area = cv2.contourArea(cnt)
    current_area = full_area
    erosion = mask
    
    # erode while the area ratio is higher than 70%
    while current_area/full_area > 0.7:
        erosion = cv2.erode(erosion,kernel,iterations = 1)
        new_cnts, hierarchy = cv2.findContours(erosion,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        newcontours = sorted(new_cnts, key=cv2.contourArea, reverse=True)
        cnt = newcontours[0][:,0]
        current_area = cv2.contourArea(cnt)
        
    return erosion

def feature_13_f14_15(img_name):
    
    print('__Color_13_14_15__ :',img_name)
    
    #pre processing, already explained in previous functions
    filename = 'images/im/{}.jpg'.format(img_name)
    m_colored = imread(filename).astype(np.uint8)

    filemask = 'images/im/{}_segmentation.jpg'.format(img_name)
    cvu8_mask = imread(filemask).astype(np.uint8)
    thresh, masku8 = cv2.threshold(cvu8_mask, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(masku8,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # inner region is the result of erosion
    inner_mask = erode_mask(masku8)
    inner_mask_expand = np.expand_dims(inner_mask, axis=2)
    inner_mask_sum = np.sum(inner_mask)
    # outer region is the source mask minus the eroded part
    outer_mask = masku8-inner_mask
    outer_mask_expand = np.expand_dims(outer_mask, axis=2)
    outer_mask_sum = np.sum(outer_mask)

    lab_img = cv2.cvtColor(m_colored, cv2.COLOR_RGB2LAB)
    inner_lab_img = (inner_mask_expand*lab_img) 
    outer_lab_img = (outer_mask_expand*lab_img)
    li, ai, bi = cv2.split(inner_lab_img )
    lo, ao, bo = cv2.split(outer_lab_img)
    f13 = np.sum(li)/inner_mask_sum - np.sum(lo)/outer_mask_sum
    f14 = np.sum(ai)/inner_mask_sum - np.sum(ao)/outer_mask_sum
    f15 = np.sum(bi)/inner_mask_sum - np.sum(bo)/outer_mask_sum
    
    return (f13,f14,f15)

###################ª###       features f16, f17 and f18     ##############################♣
    
def feature_16_17_18(img_name):
    
    print('__Color_16_17_18__ :',img_name)
    
    # usual pre processing
    filename = 'images/im/{}.jpg'.format(img_name)
    m_colored = imread(filename).astype(np.uint8)

    filemask = 'images/im/{}_segmentation.jpg'.format(img_name)
    cvu8_mask = imread(filemask).astype(np.uint8)
    thresh, masku8 = cv2.threshold(cvu8_mask, 127, 255, cv2.THRESH_BINARY)

    contours, hierarchy = cv2.findContours(masku8,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    inner_mask = erode_mask(masku8)
    outer_mask = masku8-inner_mask

    inner_valuesLAB = m_colored[inner_mask.astype(bool),:]
    outer_valuesLAB = m_colored[outer_mask.astype(bool),:]
    
    li = inner_valuesLAB[:,0]
    ai = inner_valuesLAB[:,1]
    bi = inner_valuesLAB[:,2]
    lo = outer_valuesLAB[:,0]
    ao = outer_valuesLAB[:,1]
    bo = outer_valuesLAB[:,2]
    
    x = np.arange(256)
    
    # compute gaussian kernel density estimation
    kde_li = gaussian_kde(li)
    kli = kde_li.evaluate(x)
    kde_ai = gaussian_kde(ai)
    kai = kde_ai.evaluate(x)
    kde_bi = gaussian_kde(bi)
    kbi = kde_bi.evaluate(x)
    kde_lo = gaussian_kde(lo)
    klo = kde_lo.evaluate(x)
    kde_ao = gaussian_kde(ao)
    kao = kde_ao.evaluate(x)
    kde_bo = gaussian_kde(bo)
    kbo = kde_bo.evaluate(x)
    
    # compute the overlapping area bewteen the 2 functions
    # it is approached as a sampling method is used
    f16 = np.sum(np.minimum(kli,klo))
    f17 = np.sum(np.minimum(kai,kao))
    f18 = np.sum(np.minimum(kbi,kbo))
    
    return (f16,f17,f18)