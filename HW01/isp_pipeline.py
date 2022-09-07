# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 13:45:23 2022

@author: qilin sun
"""

# Part of the codes are from OpenISP, HDR plus and Halid
# Do not cheat and post any code to the public!

from BayerDomainProcessor import *
from RGBDomainProcessor import *
from YUVDomainProcessor import *

import rawpy
import os
import sys
import time
import cv2 #only for save images to jpg
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from matplotlib import pyplot as plt #use it only for debug

import numpy as np

raw_path = 'test.RAW'
config_path = 'config.csv'
raw = rawpy.imread(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'DSC02878.ARW'))


rgb2xyz = np.asarray([[0.5767309,  0.1855540,  0.1881852],
 [0.2973769,  0.6273491,  0.0752741],
 [0.0270343,  0.0706872,  0.9911085]])

xyz2rgb =  np.asarray([[2.0413690, -0.5649464, -0.3446944],
[-0.9692660,  1.8760108,  0.0415560],
 [0.0134474, -0.1183897,  1.0154096]])

raw_h, raw_w = raw.raw_image.shape
dpc_thres = 30
dpc_mode = 'gradient'
dpc_clip = raw.camera_white_level_per_channel[0]
bl_r, bl_gr, bl_gb, bl_b =  raw.black_level_per_channel
alpha, beta = 0, 0
blackLevelCompensation_clip = raw.camera_white_level_per_channel[0]
bayer_pattern = 'rggb'
r_gain, gr_gain, gb_gain, b_gain = 1/(1996/1024), 1.0, 1.0, 1/(2080/1024)
awb_clip = raw.camera_white_level_per_channel[0]
cfa_mode = 'malvar'
cfa_clip = raw.camera_white_level_per_channel[0]
csc = np.asarray([[0.299, 0.587, 0.114], [-0.1687, -0.3313, 0.5], [ 0.5,-0.4187,-0.0813], [0, 128, 128]])
ccm = np.asarray([[1024.,    0.,    0.,    0.], [   0., 1024.,    0.,    0.], [   0.,    0., 1024.,    0.]])
edge_filter = np.asarray([[-1.,  0., -1.,  0., -1.], [-1.,  0.,  8.,  0., -1.], [-1.,  0., -1.,  0., -1.]])
ee_gain, ee_thres,  ee_emclip = [32, 128], [32, 64], [-64, 64]
fcs_edge, fcs_gain, fcs_intercept, fcs_slope = [32, 64], 32, 2, 3
nlm_h, nlm_clip = 10, 255
bnf_dw = np.asarray([[8.,12.,32.,12.,8.], [12.,64.,128.,64.,12.], [32.,128.,1024.,128.,32.], [0.,0.,0.,0.,0.], [0.,0.,0.,0.,0.]])
bnf_rw, bnf_rthres, bnf_clip = [0, 8, 16, 32], [128, 32, 8], 255
ee_gain, ee_thres, ee_emclip = [32, 128],  [32, 64], [-64, 64]
dpc_thres, dpc_mode, dpc_clip  = 30, 'gradient', 4095
fcs_edge, fcs_gain, fcs_intercept, fcs_slope = [32, 32], 32, 2, 3
nlm_h,nlm_clip = 15, 255
hue, saturation, hsc_clip, brightness, contrast, bcc_clip = 128, 256, 255, 10, 10, 255

            



rawimg = raw.raw_image 
rawimg = np.clip(rawimg, raw.black_level_per_channel[0], 2**14)
print(50*'-' + '\nLoading RAW Image Done......')
# plt.imshow(rawimg, cmap='gray')
# plt.show()

#################################################################################################################
#####################################  Part 1: Bayer Domain Processing Steps  ###################################
#################################################################################################################
t_start = time.time()
# Step 1. Dead Pixel Correction (10pts)
dpc = deadPixelCorrection(rawimg, dpc_thres, dpc_mode, dpc_clip)
Bayer_dpc = dpc.execute()
print(50*'-' + '\n 1.1 Dead Pixel Correction Done......')
cv2.imwrite('Dead Pixel Correction.jpg',Bayer_dpc)
# Step 2.'Black Level Compensation' (5pts)
parameter = raw.black_level_per_channel + [alpha, beta]
blkC = blackLevelCompensation(Bayer_dpc, parameter, bayer_pattern, clip = 2**14)
Bayer_blackLevelCompensation = blkC.execute()
print(50*'-' + '\n 1.2 Black Level Compensation Done......')
cv2.imwrite('Black Level Compensation.jpg',Bayer_blackLevelCompensation)
# Step 3.'lens shading correction
# skip this

# Step 4. Anti Aliasing Filter (10pts)
antiAliasingFilter = antiAliasingFilter(Bayer_blackLevelCompensation)
Bayer_antiAliasingFilter = antiAliasingFilter.execute()
print(50*'-' + '\n 1.4 Anti-aliasing Filtering Done......')
cv2.imwrite('Bayer_antiAliasingFilter.jpg',Bayer_antiAliasingFilter )

# Step 5. Auto White Balance and Gain Control (10pts)
parameter = [r_gain, gr_gain, gb_gain, b_gain]
awb = AWB(Bayer_blackLevelCompensation, parameter, bayer_pattern, awb_clip)
Bayer_awb = awb.execute()
print(50*'-' + '\n 1.5 White Balance Gain Done......')
cv2.imwrite('White Balance Gain.jpg',Bayer_awb)

# # Step 6. Chroma Noise Filtering (Extra 20pts)
# cnf = ChromaNoiseFiltering(Bayer_awb, bayer_pattern, 0, parameter, cfa_clip)
# Bayer_cnf = cnf.execute()
# print(50*'-' + '\n 1.6 Chroma Noise Filtering Done......')

# Step 7. 'Color Filter Array Interpolation'  Malvar (20pts)
cfa = CFA_Interpolation(Bayer_awb, cfa_mode, bayer_pattern, cfa_clip)
rgbimg_cfa = cfa.execute()
print(50*'-' + '\n 1.7 Demosaicing Done......')
cv2.imwrite('CFA_Interpolation.jpg',cv2.cvtColor(rgbimg_cfa,cv2.COLOR_RGB2BGR))
# exit()
#####################################  Bayer Domain Processing end    ###########################################
#################################################################################################################
gamma = GammaCorrection(rgbimg_cfa,None,'gamma')
rgbimg_cfa = gamma.execute()
cv2.imwrite('GammaCorrection.jpg',cv2.cvtColor(rgbimg_cfa,cv2.COLOR_RGB2BGR))
YUV = RGB2YUV(rgbimg_cfa)

#################################################################################################################
#####################################    Part 2: YUV Domain Processing Steps  ###################################
#################################################################################################################

# Step Luma-2  Edge Enhancement  for Luma (20pts)
ee = EdgeEnhancement(YUV[:,:,0], edge_filter, ee_gain, ee_thres, ee_emclip)
yuvimg_ee,yuvimg_edgemap = ee.execute()
print(50*'-' + '\n 3.Luma.2  Edge Enhancement Done......')

# Step Luma-3 Brightness/Contrast Control (5pts)
contrast = contrast / pow(2,5)    #[-32,128]
bcc = BrightnessContrastControl(yuvimg_ee, brightness, contrast, bcc_clip)
yuvimg_bcc = bcc.execute()
print(50*'-' + '\nBrightness/Contrast Adjustment Done......')
# Step Chroma-1 False Color Suppresion (10pts)
fcs = FalseColorSuppression(YUV[:,:,1:3], yuvimg_edgemap, fcs_edge, fcs_gain, fcs_intercept, fcs_slope)
yuvimg_fcs = fcs.execute()
print(50*'-' + '\n 3.Chroma.1 False Color Suppresion Done......')
# Step Chroma-2 Hue/Saturation control (10pts)
hsc = HueSaturationControl(yuvimg_fcs, hue, saturation, hsc_clip)
yuvimg_hsc = hsc.execute()
print(50*'-' + '\n 3.Chroma.2  Hue/Saturation Adjustment Done......')


# Concate Y UV Channels
yuvimg_out = np.zeros_like(YUV) 
yuvimg_out[:,:,0] = yuvimg_bcc
yuvimg_out[:,:,1:3] = yuvimg_hsc

RGB = YUV2RGB(yuvimg_out) # Pay attention to the bits

#####################################   End YUV Domain Processing Steps  ###################################
#################################################################################################################
print('ISP time cost : %.9f sec' %(time.time()-t_start)) 

 
cv2.imwrite('results.jpg',cv2.cvtColor(RGB,cv2.COLOR_RGB2BGR)) 

