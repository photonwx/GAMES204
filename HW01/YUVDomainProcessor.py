# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 20:51:14 2021

@author: qilin
"""
import numpy as np
from matplotlib import pyplot as plt #use it only for debug




# Step Luma-2. 'Edge Enhancement'
class EdgeEnhancement:
    def __init__(self, img, edge_filter, gain, thres, emclip):
        self.img = img
        self.edge_filter = edge_filter
        self.gain = gain
        self.thres = thres
        self.emclip = emclip

    def padding(self):
        img_pad = np.pad(self.img, ((1, 1), (2, 2)), 'reflect')
        print(img_pad.shape)
        return img_pad

    @staticmethod
    def clipping(img):
        np.clip(img, 0, 255, out=img)
        return img

    def emlut(self, val, thres, gain, clip):  # Edge map look up table
        lut = 0
        # Your code here 
        return lut

    # def execute(self):
    def execute(self):
        padding_img = self.padding()
        H = self.img.shape[0]
        W = self.img.shape[1]
        ee_img = np.empty((H, W), np.int16)
        em_img = np.empty((H, W), np.int16)

        P0 = padding_img[1:H+1,2:W+2] 
        P1 = padding_img[0:H,0:W] 
        P2 = padding_img[0:H,2:W+2] 
        P3 = padding_img[0:H,4:W+4] 
        P4 = padding_img[1:H+1,0:W] 
        P5 = padding_img[1:H+1,4:W+4] 
        P6 = padding_img[2:H+2,0:W] 
        P7 = padding_img[2:H+2,2:W+2] 
        P8 = padding_img[2:H+2,4:W+4]
        em_img = (8 * P0 - P1 - P2 - P3 - P4 - P5 - P6 - P7 - P8) / 8
        em_img[(em_img < -self.thres[1])|(em_img > self.thres[1])] *= self.gain[1]
        em_img[(em_img >= -self.thres[0])&(em_img <= self.thres[0])] = 0
        em_img[(em_img < -self.thres[0])&(em_img >= -self.thres[1])] *= self.gain[0]
        em_img[(em_img >= self.thres[0])&(em_img <= self.thres[1])]  *= self.gain[0]
        em_img = np.clip(em_img, self.emclip[0], self.emclip[1])
        ee_img = np.clip(self.img + em_img, 0, 255)


        return ee_img, em_img


# Step Luma-3. 'Brightness Contrast Control'
class BrightnessContrastControl: 
    def __init__(self, img, brightness, contrast, clip):
        self.img = img
        self.brightness = 5
        self.contrast = 1
        self.clip = clip

    def clipping(self):
        np.clip(self.img, 0, self.clip, out=self.img)
        return self.img

    def execute(self):
        # Your code here 
        self.img = self.img + self.brightness
        self.img = self.img + (self.img - 127) * self.contrast
        self.img = self.clipping()

        return self.img



# Step Chroma-1 False Color Suppression
class FalseColorSuppression:
    def __init__(self, img, edgemap, fcs_edge, gain, intercept, slope):
        self.img = img.astype(edgemap.dtype)
        self.edgemap = edgemap
        self.fcs_edge = [32,64]
        self.gain = gain
        self.intercept = intercept
        self.slope = slope

    def clipping(self):
        # Your code here 

        return  

    def execute(self):
        ## This answer if from Liu Wen, one of our classmates 
        # Your code here 
        em = np.abs(self.edgemap)
        self.img[ em > self.fcs_edge[1]] = 0
        self.img[(em > self.fcs_edge[0]) & ( em <= self.fcs_edge[1])] *= 1-(self.img[(em > self.fcs_edge[0])&(em <= self.fcs_edge[1])] - self.fcs_edge[0])\
                                                                    /(self.fcs_edge[1] - self.fcs_edge[0])

        return  self.img
    
# Step Chroma-2 Hue/Saturation control
class HueSaturationControl:
    def __init__(self, img, hue, saturation, clip):
        self.img = img
        self.hue = 0
        self.saturation = 5
        self.clip = clip

    def clipping(self):
        np.clip(self.img, 0, self.clip, out=self.img)
        return self.img

    def lut(self):
        ind = np.array([i for i in range(360)])
        sin = np.sin(ind * np.pi / 180) * 256
        cos = np.cos(ind * np.pi / 180) * 256
        lut_sin = dict(zip(ind, [round(sin[i]) for i in ind]))
        lut_cos = dict(zip(ind, [round(cos[i]) for i in ind]))
        return lut_sin, lut_cos

    def execute(self):
        ## This answer if from Liu Wen, one of our classmates 
        # Your code here 
        self.hue = self.hue * np.pi / 180
        self.img = np.einsum('ijk,kl->ijl',self.img,np.array([[np.cos(self.hue),-np.sin(self.hue)],[np.sin(self.hue),np.cos(self.hue)]]))
        self.img *= self.saturation
        
        return self.img