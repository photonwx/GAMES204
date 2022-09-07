# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 19:45:23 2021

@author: qilin sun
"""
import numpy as np


# Step 1. 'Color Correction Matrix'
class ColorCorrectionMatrix:
    def __init__(self, img, ccm):
        self.img = img
        self.ccm = ccm

    def execute(self):
        img_h = self.img.shape[0]
        img_w = self.img.shape[1]
        img_c = self.img.shape[2]
        ccm_img = np.empty((img_h, img_w, img_c), np.uint32)
        for y in range(img_h):
            for x in range(img_w):
                mulval = self.ccm[0:3,:] * self.img[y,x,:]
                ccm_img[y,x,0] = np.sum(mulval[0]) + self.ccm[3,0]
                ccm_img[y,x,1] = np.sum(mulval[1]) + self.ccm[3,1]
                ccm_img[y,x,2] = np.sum(mulval[2]) + self.ccm[3,2]
                ccm_img[y,x,:] = ccm_img[y,x,:]  
        self.img = ccm_img
        return self.img

# Step 2. 'Gamma Correction'
class GammaCorrection:
    def __init__(self, img, lut, mode):
        self.img = img
        self.lut = lut
        # Here you can choose to cread a look up table(LUT) saved from Photoshop, you will get a better results and higher grades.
        # Or simply apply a gamma if LUT is too hard for you.
        self.mode = mode

    def execute(self):
        #gamma correction
        if self.mode == 'gamma':
            self.img = self.img.astype(np.float32)
            self.img /= np.max(self.img)
            self.img = self.img ** (1/2.5) * 255

        return self.img

# Step 3 . Color Space Conversion   RGB-->YUV

def RGB2YUV(img):
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    # Your code here
    Y = 0.299*R + 0.587*G + 0.114*B
    U = -0.147*R - 0.289*G + 0.436*B
    V = 0.615*R - 0.515*G - 0.100*B

    return np.stack([Y,U,V], axis=2)
def YUV2RGB(img):
    Y = img[:,:,0]
    U = img[:,:,1]
    V = img[:,:,2]
    # Your code here
    R = Y + 1.13983*V
    G = Y - 0.39465*U - 0.58060*V
    B = Y + 2.03211*U

    return np.stack([R,G,B], axis=2)