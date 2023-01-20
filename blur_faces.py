# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 01:23:27 2023

@author: Muniza
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('DATA/Nadia_Murad.jpg',0) 
solvay = cv2.imread('DATA/solvay_conference.jpg',0)

#Load haar cascade
face_cascade = cv2.CascadeClassifier('DATA/haarcascades/haarcascade_frontalface_default.xml')

#Function to display images
def display(img):
    fig = plt.figure(figsize=(10,8)) #resize image
    ax = fig.add_subplot(111)
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #color correct
    ax.imshow(new_img)
    
#Blur out plate
def detect_and_blur_face(img):
    
    # img = solvay.copy()
    face_img = img.copy()
    roi = img.copy() #Region of Interest = face
    
    #Find faces
    face_rects = face_cascade.detectMultiScale(face_img,scaleFactor=1.2, minNeighbors=5)
    
    (H,W) = img.shape[:2] #get height width of img
    #determine size of blurring kernel
    kW = int(W/3)
    if kW % 2 == 0:
        kW -= 1
        
    kH = int(H/3)
    if kH % 2 == 0:
        kH -= 1
        
    i = 0
    for (x0,y0,w,h) in face_rects:
        x1 = x0 + w
        y1 = y0 + h
        if x1 > W:
            x1 = W
        if y1 > H:
            y1 = H

        i += 1
        roi = face_img[y0:y1,x0:x1] #define roi using face_rects detection
        blurred_roi = cv2.GaussianBlur(roi,(kW,kH),0) #apply blur to roi only
        face_img[y0:y1,x0:x1] = blurred_roi #set roi to blurred roi on img to display
        
    # del i,x,y,w,h,face_rects
    return face_img   

result = detect_and_blur_face(solvay)
display(result)
