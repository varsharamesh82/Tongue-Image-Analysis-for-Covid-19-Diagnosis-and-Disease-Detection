#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import cv2
# load image and shrink - it's massive
num = 6
# 빛의 밝기에 따라 조정하세요
light = 0
img = cv2.imread('/Users/sanjana/Desktop/IP Paper/tongueimgs/24.jpg')
img = cv2.resize(img, (580, 580))
cv2.imshow('original',img)
height, width = img.shape[:2]
print('width: ', width, ' height: ', height)

threshold = 30
point = []
temp = []
flag1, flag2, flag3, flag4 = False, False, False, False
center_width = 0
part1_height = 0
part2_height = 0

# find point 1
for i in range(height):
    for j in range(width):
        B = img.item(i, j, 0)
        G = img.item(i, j, 1)
        R = img.item(i, j, 2)
        if B+G+R > threshold and j < width/2:
            flag1 = True
            temp.append([j,i])
            break
    if flag1 == True:
        break

for i in range(width):
    for j in range(height):
        B = img.item(j, i, 0)
        G = img.item(j, i, 1)
        R = img.item(j, i, 2)
        if B+G+R > threshold:
            flag2 = True
            temp.append([i,j])
            break
    if flag2 == True:
        break


i = temp[1][0]
j = temp[0][1]
while True:
    B = img.item(j, i, 0)
    G = img.item(j, i, 1)
    R = img.item(j, i, 2)
    if B+G+R > threshold:
        point.append([i,j])
        break
    i += 1
    j += 1
    
temp = []

# find point 2
for i in range(height):
    for j in range(width-1, -1, -1):
        B = img.item(i, j, 0)
        G = img.item(i, j, 1)
        R = img.item(i, j, 2)
        if B+G+R > threshold and j > width/2:
            flag3 = True
            temp.append([j,i])
            break
    if flag3 == True:
        break

for i in range(width-1, -1, -1):
    for j in range(height):
        B = img.item(j, i, 0)
        G = img.item(j, i, 1)
        R = img.item(j, i, 2)
        if B+G+R > threshold:
            flag4 = True
            temp.append([i,j])
            break
    if flag4 == True:
        break

i = temp[1][0]
j = temp[0][1]
while True:
    B = img.item(j, i, 0)
    G = img.item(j, i, 1)
    R = img.item(j, i, 2)
    if B+G+R > threshold:
        point.append([i,j])
        break
    i -= 1
    j += 1

center_width = (point[0][0]+point[1][0])//2
flag1, flag2, flag3, flag4 = False, False, False, False

# find point 3
for i in range(height):
    B = img.item(i, center_width, 0)
    G = img.item(i, center_width, 1)
    R = img.item(i, center_width, 2)
    if B+G+R > threshold:
        point.append([center_width,i])
        break

# find point 4
for i in range(height-1, -1, -1):
    B = img.item(i, center_width, 0)
    G = img.item(i, center_width, 1)
    R = img.item(i, center_width, 2)
    if B+G+R > threshold:
        point.append([center_width,i])
        break

part1_height = (point[3][1]-point[2][1])//3+point[2][1]
part2_height = (point[3][1]-point[2][1])//3*2+point[2][1]
part3_height = (part2_height + part1_height)//2
part4_height = (part3_height - part1_height) + part2_height
part5_height = (part3_height - part1_height)//2 + part4_height

# find point 5
for i in range(width):
    B = img.item(part1_height, i, 0)
    G = img.item(part1_height, i, 1)
    R = img.item(part1_height, i, 2)
    if B+G+R > threshold:
        point.append([i,part1_height])
        break

# find point 6
for i in range(width-1, -1, -1):
    B = img.item(part1_height, i, 0)
    G = img.item(part1_height, i, 1)
    R = img.item(part1_height, i, 2)
    if B+G+R > threshold:
        point.append([i,part1_height])
        break

# find point 7
for i in range(width):
    B = img.item(part3_height, i, 0)
    G = img.item(part3_height, i, 1)
    R = img.item(part3_height, i, 2)
    if B+G+R > threshold:
        point.append([i,part3_height])
        break

# find point 8
for i in range(width-1, -1, -1):
    B = img.item(part3_height, i, 0)
    G = img.item(part3_height, i, 1)
    R = img.item(part3_height, i, 2)
    if B+G+R > threshold:
        point.append([i,part3_height])
        break

# find point 9
for i in range(width):
    B = img.item(part2_height, i, 0)
    G = img.item(part2_height, i, 1)
    R = img.item(part2_height, i, 2)
    if B+G+R > threshold:
        point.append([i,part2_height])
        break

# find point 10
for i in range(width-1, -1, -1):
    B = img.item(part2_height, i, 0)
    G = img.item(part2_height, i, 1)
    R = img.item(part2_height, i, 2)
    if B+G+R > threshold:
        point.append([i,part2_height])
        break

# find point 11
for i in range(width):
    B = img.item(part4_height, i, 0)
    G = img.item(part4_height, i, 1)
    R = img.item(part4_height, i, 2)
    if B+G+R > threshold:
        point.append([i,part4_height])
        break

# find point 12
for i in range(width-1, -1, -1):
    B = img.item(part4_height, i, 0)
    G = img.item(part4_height, i, 1)
    R = img.item(part4_height, i, 2)
    if B+G+R > threshold:
        point.append([i,part4_height])
        break


# find point 13
for i in range(width):
    B = img.item(part5_height, i, 0)
    G = img.item(part5_height, i, 1)
    R = img.item(part5_height, i, 2)
    if B+G+R > threshold:
        point.append([i,part5_height])
        break

# find point 14
for i in range(width-1, -1, -1):
    B = img.item(part5_height, i, 0)
    G = img.item(part5_height, i, 1)
    R = img.item(part5_height, i, 2)
    if B+G+R > threshold:
        point.append([i,part5_height])
        break




for i, pnt in enumerate(point):
    if (i+1)%2 == 0: val = pnt[0]-20
    else: val = pnt[0]
    #cv2.putText(img, str(i+1), (val,pnt[1]-5), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (255,255,255), 2)
    #cv2.circle(img, (pnt[0], pnt[1]), 2, (255,255,255), 4)

#cv2.line(img, (point[4][0], point[4][1]), (point[5][0], point[5][1]),  (255,255,255),2)
center_point = ((point[0][0]+point[1][0]+point[3][0])//3, (point[0][1]+point[1][1]+point[3][1])//3)
#cv2.circle(img, center_point, 2, (255,255,255), 4)

inside_point = []
point_list = [6,7,8,9,10,11,12,13,3]
length = (point[5][0] - point[4][0]) * 0.15
inside_point.append([int(point[4][0]+length), point[4][1]])
inside_point.append([int(point[5][0]-length), point[5][1]])
for i in point_list:
    long_length = ((center_point[0]-point[i][0])**2 +(center_point[1]-point[i][1])**2) ** 0.5
    rate = length/long_length
    inside_point.append( [int((center_point[0]-point[i][0]) * rate + point[i][0]) , int((center_point[1]-point[i][1]) * rate + point[i][1])] )

pts = np.array([[inside_point[0][0],inside_point[0][1]], [inside_point[2][0],inside_point[2][1]], [inside_point[4][0],inside_point[4][1]], [inside_point[6][0],inside_point[6][1]],  [inside_point[8][0],inside_point[8][1]], [inside_point[-1][0], inside_point[-1][1]],  [inside_point[9][0],inside_point[9][1]],  [inside_point[7][0], inside_point[7][1]], [inside_point[5][0], inside_point[5][1]],[inside_point[3][0], inside_point[3][1]],  [inside_point[1][0], inside_point[1][1]]  ], np.int32)
pts = pts.reshape((-1, 1, 2))
#img = cv2.polylines(img, [pts], True, (255,255,255), 2)
#cv2.line(img, (point[10][0], point[10][1]), (inside_point[6][0], inside_point[6][1]),  (255,255,255),2)
#cv2.line(img, (point[11][0], point[11][1]), (inside_point[7][0], inside_point[7][1]),  (255,255,255),2)

#find coated tongue area(origin color)
pts = np.array([[point[3][0],point[3][1]], [point[12][0],point[12][1]], [point[13][0], point[13][1]]])
## (1) Crop the bounding rect
rect = cv2.boundingRect(pts)
x,y,w,h = rect
croped = img[y:y+h, x:x+w].copy()

## (2) make mask
pts = pts - pts.min(axis=0)

mask = np.zeros(croped.shape[:2], np.uint8)
cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

## (3) do bit-op
dst = cv2.bitwise_and(croped, croped, mask=mask)
dst_height, dst_width = dst.shape[:2]
min_value = 0
min_B = 0
min_G = 0
min_R = 0
for i in range(dst_height):
    for j in range(dst_width):
        B = dst.item(i, j, 0)
        G = dst.item(i, j, 1)
        R = dst.item(i, j, 2)
        if B+G+R > min_value and B+G+R < 720:
            min_B = B
            min_G = G
            min_R = R
            min_value = min_B + min_G + min_R
print([min_B, min_G, min_R])

# find coated tongue area
color = [min_B, min_G, min_R]
pixel = np.uint8([[color]])
pixel2 = np.uint8([[[0,0,255]]])
hsv = cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)
hsv2 = cv2.cvtColor(pixel2, cv2.COLOR_BGR2HSV)
print(hsv)
print(hsv2)

#lower_hsv = (int(hsv[0][0][1])-50, int(hsv[0][0][1])-50, int(hsv[0][0][2])-50)
lower_hsv = (0, int(hsv[0][0][1])-light, 100)
upper_hsv = (179, 255, 255)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('img_hsv', img_hsv)
img_mask = cv2.inRange(img_hsv, lower_hsv, upper_hsv)
print(img_mask)
img_result = cv2.bitwise_and(img, img, mask = img_mask)
cv2.imshow('img_result', img_result)
cv2.imwrite('/Users/sanjana/Desktop/IP Paper/tonguehsv.jpg', img_hsv)
cv2.imwrite('/Users/sanjana/Desktop/IP Paper/tongueres.jpg', img_result)
cv2.imshow('img', img)
#cv2.imwrite('C:/Users/SungHyeon/Desktop/T/'+str(num)+'_point.jpg',img)

cv2.waitKey(0)
cv2.destroyAllWindows()
