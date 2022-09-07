# -*- coding: utf-8 -*-
"""
Created on Fri Aug 27 19:45:23 2021

@author: qilin sun
"""
from turtle import shape
import numpy as np
# For this file, use numpy only.  Try your best to get the best visual pleasent
# result, as well as the fasest speed and smallest memory consumption.

# Step 1. Dead Pixel Correction (10pts)
class deadPixelCorrection:
    
    def __init__(self, img, thres, mode, clip):
        self.img = img
        self.thres = thres
        self.mode = mode
        self.clip = clip
    
    def padding(self):
        # padding needed for avoid black boundry
        # Fill your code here
        padding_img = np.pad(self.img, (2, 2), 'reflect')
        return padding_img

    def clipping(self):
        # clip needed for avoid values>maximum
        # Fill your code here
        np.clip(self.img, 0, self.clip, out=self.img)
        return self.img

    def execute(self):
        # Fill your code here
        H,W = self.img.shape[0],self.img.shape[1]
        dpc_img = np.zeros((H,W),np.uint16)
        padding_img = self.padding()  
        P0 = padding_img[2:H+2,2:W+2] 
        P1 = padding_img[0:H,0:W] 
        P2 = padding_img[0:H,2:W+2] 
        P3 = padding_img[0:H,4:W+4] 
        P4 = padding_img[2:H+2,0:W] 
        P5 = padding_img[2:H+2,4:W+4] 
        P6 = padding_img[4:H+4,0:W] 
        P7 = padding_img[4:H+4,2:W+2] 
        P8 = padding_img[4:H+4,4:W+4] 
        P = np.stack([P1,P2,P3,P4,P5,P6,P7,P8])
        # print(np.min(np.abs(P-P0)))
        dead_pixels = np.min(np.abs(P-P0),axis=0) > self.thres 
        if self.mode == 'mean':
            P_all = (P2 + P4 + P5 + P7) // 4
            P0[dead_pixels] = P_all[dead_pixels]
        elif self.mode == 'gradient':
            dv = abs(2 * P0 - P2 - P7 )
            dh = abs(2 * P0 - P4  - P5 )
            ddl = abs(2 * P0 - P1  - P8 )
            ddr = abs(2 * P0 - P3  - P6 )

            P_all = np.stack([dv, dh, ddl, ddr])

            P_dv = np.min(P_all,axis=0) == dv
            P_dh = np.min(P_all,axis=0) == dh
            P_ddl = np.min(P_all,axis=0) == ddl
            P_ddr = np.min(P_all,axis=0) == ddr

            P_dv = P_dv * dead_pixels
            P_dh = P_dh * dead_pixels
            P_ddl  = P_ddl  * dead_pixels
            P_ddr = P_ddr * dead_pixels
            
            P0[P_dv ] = (P2[P_dv] + P7[P_dv] + 1) / 2
            P0[P_dh ] =  (P4[P_dh] + P5[P_dh] + 1) / 2
            P0[P_ddl ] = (P1[P_ddl] + P8[P_ddl] + 1) / 2
            P0[P_ddr ] =  (P3[P_ddr] + P6[P_ddr] + 1) / 2
        dpc_img = P0
        self.img = dpc_img
        self.img = self.clipping()
        return self.img
    


# Step 2.'Black Level Compensation'   (10pts)
class blackLevelCompensation:
    def __init__(self, img, parameter, bayer_pattern = 'rggb', clip=255):
        self.img = img
        self.parameter = parameter
        self.bayer_pattern = bayer_pattern
        self.clip = clip

    def clipping(self):
        # clip needed for avoid values>maximum, find a proper value for 14bit raw input
        # Fill your code here
        np.clip(self.img, 0, self.clip, out=self.img)
        return self.img 

    def execute(self):
        bl_r = -self.parameter[0]
        bl_gr = -self.parameter[1]
        bl_gb = -self.parameter[2]
        bl_b = -self.parameter[3]
        alpha = self.parameter[4]
        beta = self.parameter[5]
        # Fill your code here
        H,W = self.img.shape[0],self.img.shape[1]
        blc_img = np.zeros((H,W),np.uint16)
        if self.bayer_pattern == 'rggb':
            r = self.img[0::2, 0::2] + bl_r
            b = self.img[1::2, 1::2] + bl_b
            gr = self.img[0::2, 1::2] + bl_gr + alpha * r 
            gb = self.img[1::2, 0::2] + bl_gb + beta * b 
            
            blc_img[0::2, 0::2] = r
            blc_img[0::2, 1::2] = gr
            blc_img[1::2, 0::2] = gb
            blc_img[1::2, 1::2] = b
        self.img = blc_img
        self.img = self.clipping()
        
        return self.img 

# Step 3.'lens shading correction  
# Skip this step

# Step 4. Anti Aliasing Filter (10pts)
class antiAliasingFilter:
    'Anti-aliasing Filter'
    def __init__(self, img):
        self.img = img

    def padding(self):
        # padding needed for avoid black boundry
        # Fill your code here
        padding_img = np.pad(self.img, (2, 2), 'reflect')
        return padding_img
 
    # Hint: "In bayer domain, the each ,R,G,G,B pixel is skipped by 2."
    def execute(self):
        # Fill your code here
        H,W = self.img.shape[0],self.img.shape[1]
        # aaf_img = np.zeros((H,W),np.uint16)
        padding_img = self.padding()  

        P0 = padding_img[2:H+2,2:W+2] 
        P1 = padding_img[0:H,0:W] 
        P2 = padding_img[0:H,2:W+2] 
        P3 = padding_img[0:H,4:W+4] 
        P4 = padding_img[2:H+2,0:W] 
        P5 = padding_img[2:H+2,4:W+4] 
        P6 = padding_img[4:H+4,0:W] 
        P7 = padding_img[4:H+4,2:W+2] 
        P8 = padding_img[4:H+4,4:W+4] 
        aaf_img = (8 * P0 + P1 + P2 + P3 + P4 + P5 + P6 + P7 + P8 ) / 16
        return aaf_img

# Step 5. Auto White Balance and Gain Control (10pts)
class AWB:
    def __init__(self, img, parameter, bayer_pattern, clip):
        self.img = img
        self.parameter = parameter
        self.bayer_pattern = bayer_pattern
        self.clip = clip

    def clipping(self):
        # clip needed for avoid values>maximum, find a proper value for 14bit raw input
        # Fill your code here
        np.clip(self.img, 0, self.clip, out=self.img)
        return self.img 

    def execute(self):
        # calculate Gr_avg/R_avg, 1, Gr_avg/Gb_avg, Gr_avg/B_avg and apply to each channel
        # Fill your code here

        H,W = self.img.shape[0],self.img.shape[1]
        awb_img = np.zeros((H,W))
        if self.bayer_pattern == 'rggb':

            r = self.img[0::2, 0::2] 
            gr = self.img[0::2, 1::2] 
            gb = self.img[1::2, 0::2] 
            b = self.img[1::2, 1::2] 
            r_gain = np.mean(gr) / np.mean(r)
            gr_gain = 1 
            gb_gain =  np.mean(gr) / np.mean(gb)
            b_gain =  np.mean(gr) / np.mean(b)
            awb_img[0::2, 0::2] = r * r_gain
            awb_img[0::2, 1::2] = gr * gr_gain
            awb_img[1::2, 0::2] = gb * gb_gain 
            awb_img[1::2, 1::2] = b * b_gain
        self.img = awb_img
        self.img = self.clipping()
        return self.img 
    
# Step 6. Chroma Noise Reduction (Additional 20pts)
# Ref: https://patentimages.storage.googleapis.com/a8/b7/82/ef9d61314d91f6/US20120237124A1.pdf

class ChromaNoiseFiltering:
    def __init__(self, img, bayer_pattern, thres, gain, clip):
        self.img = img
        self.bayer_pattern = bayer_pattern
        self.thres = thres
        self.gain = gain
        self.clip = clip

    def padding(self):
        # Fill your code here
        padding_img = np.pad(self.img, (2, 2), 'reflect')
        return padding_img

    def clipping(self):
        # Fill your code here
        np.clip(self.img, 0, self.clip, out=self.img)
        return self.img  

    def cnc(self, is_color, center, avgG, avgC1, avgC2):
        'Chroma Noise Correction'
        r_gain, gr_gain, gb_gain, b_gain = self.gain[0], self.gain[1], self.gain[2], self.gain[3]
        dampFactor = 1.0
        signalGap = center - max(avgG, avgC2)
        # Fill your code here
        if is_color == 'r':
            if r_gain <= 1.0:
                dampFactor = 1.0
            elif r_gain > 1.0 and r_gain <= 1.2:
                dampFactor = 0.5
            elif r_gain > 1.2:
                dampFactor = 0.3
        elif is_color == 'b':
            if b_gain <= 1.0:
                dampFactor = 1.0
            elif b_gain > 1.0 and b_gain <= 1.2:
                dampFactor = 0.5
            elif b_gain > 1.2:
                dampFactor = 0.3
        chromaCorrected = max(avgG, avgC2) + dampFactor * signalGap
        if is_color == 'r':
            signalMeter = 0.299 * avgC1 + 0.587 * avgG + 0.114 * avgC2
        elif is_color == 'b':
            signalMeter = 0.299 * avgC2 + 0.587 * avgG + 0.114 * avgC1
        if signalMeter <= 30:
            fade1 = 1.0
        elif signalMeter > 30 and signalMeter <= 50:
            fade1 = 0.9
        elif signalMeter > 50 and signalMeter <= 70:
            fade1 = 0.8
        elif signalMeter > 70 and signalMeter <= 100:
            fade1 = 0.7
        elif signalMeter > 100 and signalMeter <= 150:
            fade1 = 0.6
        elif signalMeter > 150 and signalMeter <= 200:
            fade1 = 0.3
        elif signalMeter > 200 and signalMeter <= 250:
            fade1 = 0.1
        else:
            fade1 = 0
        if avgC1 <= 30:
            fade2 = 1.0
        elif avgC1 > 30 and avgC1 <= 50:
            fade2 = 0.9
        elif avgC1 > 50 and avgC1 <= 70:
            fade2 = 0.8
        elif avgC1 > 70 and avgC1 <= 100:
            fade2 = 0.6
        elif avgC1 > 100 and avgC1 <= 150:
            fade2 = 0.5
        elif avgC1 > 150 and avgC1 <= 200:
            fade2 = 0.3
        elif avgC1 > 200:
            fade2 = 0
        fadeTot = fade1 * fade2
        center_out = (1 - fadeTot) * center + fadeTot * chromaCorrected
        return center_out

    def cnd(self, y, x, img):
        'Chroma Noise Detection'
        avgG = 0
        avgC1 = 0
        avgC2 = 0
        is_noise = 0
        # Fill your code here
        for i in range(y - 4, y + 4, 1):
            for j in range(x - 4, x + 4, 1):
                if i % 2 == 1 and j % 2 == 0:
                    avgG = avgG + img[i,j]
                elif i % 2 == 0 and j % 2 == 1:
                    avgG = avgG + img[i, j]
                elif i % 2 == 0 and j % 2 == 0:
                    avgC1 = avgC1 + img[i,j]    # weights are equal, could be as gaussian dist
                elif i % 2 == 1 and j % 2 == 1:
                    avgC2 = avgC2 + img[i,j]
        avgG = avgG / 40
        avgC1 = avgC1 / 25
        avgC2 = avgC2 / 16
        center = img[y, x]
        if center > avgG + self.thres and center > avgC2 + self.thres:
            if avgC1 > avgG + self.thres and avgC1 > avgC2 + self.thres:
                is_noise = 1
            else:
                is_noise = 0
        else:
            is_noise = 0
        return is_noise, avgG, avgC1, avgC2

    def cnf(self, is_color, y, x, img):
        is_noise, avgG, avgC1, avgC2 = self.cnd(y, x, img)
        # Fill your code here
        if is_noise:
            pix_out = self.cnc(is_color, img[y,x], avgG, avgC1, avgC2)
        else:
            pix_out = img[y,x]
        return pix_out


    def execute(self):
        # Fill your code here
        img_pad = self.padding()
        raw_h = self.img.shape[0]
        raw_w = self.img.shape[1]
        cnf_img = np.empty((raw_h, raw_w), np.uint16)
        for y in range(0, img_pad.shape[0] - 8 - 1, 2):
            for x in range(0, img_pad.shape[1] - 8 - 1, 2):
                if self.bayer_pattern == 'rggb':
                    r = img_pad[y + 4, x + 4]
                    gr = img_pad[y + 4, x + 5]
                    gb = img_pad[y + 5, x + 4]
                    b = img_pad[y + 5, x + 5]
                    cnf_img[y, x] = self.cnf('r', y + 4, x + 4, img_pad)
                    cnf_img[y, x + 1] = gr
                    cnf_img[y + 1, x] = gb
                    cnf_img[y + 1, x + 1] = self.cnf('b', y + 5, x + 5, img_pad)
        self.img = cnf_img
        return self.clipping()


# Step 7. 'Color Filter Array Interpolation'  with Malvar Algorithm ”High Quality Linear“ (20pts)
class CFA_Interpolation:
    def __init__(self, img, mode, bayer_pattern, clip):
        self.img = img
        self.mode = mode
        self.bayer_pattern = bayer_pattern
        self.clip = clip

    def padding(self):
        # Fill your code here
        img_pad = np.pad(self.img, ((2,2),(2,2)), 'reflect')

        return img_pad

    def clipping(self):
        # Fill your code here

        return 


    def execute(self):

        ## This answer if from Liu Wen, one of our classmates 
        img_pad = self.padding()
        img_pad = img_pad.astype(np.int32)
        raw_h = self.img.shape[0]
        raw_w = self.img.shape[1]
        cfa_img = np.empty((raw_h, raw_w, 3), np.int16)
        r = np.zeros((self.img.shape))
        g = np.zeros((self.img.shape))
        b = np.zeros((self.img.shape))
        

        G_at_R = G_at_B = np.array([[0,0,-1,0,0],[0,0,2,0,0],[-1,2,4,2,-1],[0,0,2,0,0],[0,0,-1,0,0]])
        R_row_at_G = B_row_at_G = np.array([[0,0,1/2,0,0],[0,-1,0,-1,0],[-1,4,5,4,-1],[0,-1,0,-1,0],[0,0,1/2,0,0]])
        R_col_at_G = B_col_at_G = R_row_at_G.T
        R_at_B = B_at_R = np.array([[0,0,-3/2,0,0],[0,2,0,2,0],[-3/2,0,6,0,-3/2],[0,2,0,2,0],[0,0,-3/2,0,0]])

        r[::2,::2] = img_pad[2:-2:2,2:-2:2]
        for i in range(-2,3):
            for j in range(-2,3):
                if R_at_B[i+2,j+2] != 0:
                    r[1::2,1::2] += (img_pad[3+i:,3+j:]*R_at_B[i+2,j+2])[:raw_h:2,:raw_w:2]
                if R_col_at_G[i+2,j+2] != 0:
                    r[1::2,::2] += (img_pad[3+i:,2+j:]*R_col_at_G[i+2,j+2])[:raw_h:2,:raw_w:2]
                if R_row_at_G[i+2,j+2] != 0:
                    r[::2,1::2] += (img_pad[2+i:,3+j:]*R_row_at_G[i+2,j+2])[:raw_h:2,:raw_w:2]
        r[1::2,::2] /= 8
        r[::2,1::2] /= 8
        r[1::2,1::2] /= 8

        g[1::2,::2] = img_pad[3:-1:2,2:-2:2]
        g[::2,1::2] = img_pad[2:-2:2,3:-1:2]
        for i in range(-2,3):
            for j in range(-2,3):
                if G_at_B[i+2,j+2] != 0:
                    g[1::2,1::2] += (img_pad[3+i:,3+j:]*G_at_B[i+2,j+2])[:raw_h:2,:raw_w:2]
                if G_at_R[i+2,j+2] != 0:
                    g[::2,::2] += (img_pad[2+i:,2+j:]*G_at_R[i+2,j+2])[:raw_h:2,:raw_w:2]
        g[::2,::2] /= 8
        g[1::2,1::2] /= 8

        b[1::2,1::2] = img_pad[3:-1:2,3:-1:2]
        for i in range(-2,3):
            for j in range(-2,3):
                if B_at_R[i+2,j+2] != 0:
                    b[::2,::2] += (img_pad[2+i:,2+j:]*B_at_R[i+2,j+2])[:raw_h:2,:raw_w:2]
                if B_row_at_G[i+2,j+2] != 0:
                    b[1::2,::2] += (img_pad[3+i:,2+j:]*B_row_at_G[i+2,j+2])[:raw_h:2,:raw_w:2]
                if B_col_at_G[i+2,j+2] != 0:
                    b[::2,1::2] += (img_pad[2+i:,3+j:]*B_col_at_G[i+2,j+2])[:raw_h:2,:raw_w:2]
        b[::2,::2] /= 8
        b[1::2,::2] /= 8
        b[::2,1::2] /= 8
        
    
        cfa_img = np.stack([r,g,b],axis=2)
        cfa_img = np.clip(cfa_img, 0, 2**14)
        cfa_img = (np.clip(cfa_img/2000,0,1)*255).astype(np.uint8)

        return cfa_img
