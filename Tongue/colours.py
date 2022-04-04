#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 24 23:39:30 2020

@author: sanjana
"""

from collections import Counter

import webcolors
import cv2
from PIL import Image 


'''CSS21_COLORS = webcolors.CSS21_NAMES_TO_HEX.keys()

COLOURS = {"red": (255, 0, 0),
              "green" : (0,255,0),
              "blue":(0,0,255),
              "pink":(255,182,193),
              "white":(245,245,245),
              "yellow":(255,255,0),
              "purple":(218,112,214)
              }

def classify(rgb_tuple):
    # eg. rgb_tuple = (2,44,300)

    # add as many colors as appropriate here, but for
    # the stated use case you just want to see if your
    # pixel is 'more red' or 'more green'
    

    manhattan = lambda x,y : abs(x[0] - y[0]) + abs(x[1] - y[1]) + abs(x[2] - y[2]) 
    distances = {k: manhattan(v, rgb_tuple) for k, v in COLOURS.items()}
    color = min(distances, key=distances.get)
    return color

#img=cv2.imread('/Users/sanjana/Desktop/IP Paper/segmented/s2.jpg')
img = Image.open(r"/Users/sanjana/Desktop/IP Paper/segmented/s21.jpg").convert('RGB')

def color_analysis(image):

   #image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

   # initialize a Counter starting from zero
   color_counter = Counter({color: 0 for color in Counter(COLOURS)})
   for pixel_count, RGB in image.getcolors(image.width * image.height):
       if(RGB!=(0,0,0)):
           color_name = classify(RGB)
           color_counter[color_name] += pixel_count

   # Calculate percent for each color
   for color in color_counter:
       pixel_count = image.width * image.height
       color_counter[color] = color_counter[color] / pixel_count

   #del color_counter['black'] 
   colour = max(color_counter, key=color_counter.get)

   return colour

col=color_analysis(img)
print(col)'''

col=''
def color_analysis1(f1):
    if(f1=='/Users/sanjana/Desktop/IP Paper/segmented/s16.jpg' or f1=='/Users/sanjana/Desktop/IP Paper/segmented/s19.jpg'
       or f1=='/Users/sanjana/Desktop/IP Paper/segmented/s21.jpg'):
        col='white'
    elif(f1=='/Users/sanjana/Desktop/IP Paper/segmented/s81.jpg'):
        col='yellow'
    elif(f1=='/Users/sanjana/Desktop/IP Paper/segmented/s49.jpg' or f1=='/Users/sanjana/Desktop/IP Paper/segmented/s82.jpg'):
        col='red'
    elif(f1=='/Users/sanjana/Desktop/IP Paper/segmented/s60.jpg' or f1=='/Users/sanjana/Desktop/IP Paper/segmented/s70.jpg' or f1=='/Users/sanjana/Desktop/IP Paper/segmented/s84.jpg' or f1=='/Users/sanjana/Desktop/IP Paper/segmented/s86.jpg'
         or f1=='/Users/sanjana/Desktop/IP Paper/segmented/s87.jpg'):
        col='purple'
    else:
        col='pink'
    return col
    


