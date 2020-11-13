# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 20:02:10 2020

@author: HP
"""

import numpy as np
import heapq
from PIL import Image
import os

import matplotlib.pyplot as plt
from numpy import linalg as LA
from math import sqrt
from itertools import chain
from skimage.segmentation import mark_boundaries, find_boundaries

os.chdir("C:/Users/HP/Downloads")


"""
initializeCentroid change
updateDistance
check parameters of each definition
"""
class element:
    """ key: distance
        val: [x(X,Y), c(l,a,b), k]
    """
    def __init__(self,key,val):
        self.key = key
        self.val = val
    
    def __lt__(self, other):
        return self.key < other.key   

def neighbours(curr_pos, image_size):
    nbh = []

    x = curr_pos[0]
    y = curr_pos[1]

    if x-1 >= 0:
        nbh.append([x-1, y])

    if y-1 >= 0:
        nbh.append([x, y-1])

    if x+1 < image_size[0]:
        nbh.append([x+1, y])

    if y+1 < image_size[1]:
        nbh.append([x, y+1])

    return nbh
    
def initializeCentroid(image_size, num_of_clusters):
    """output list of centroids(pixel)"""
    """image_size = image.shape"""

    image_size_y = image_size[0] #321
    image_size_x = image_size[1]  #481
    #print(image_size_x)
    #print(image_size_y)

    # compute grid size
    num_sqr = sqrt(num_of_clusters)
    
    #image_ratio = image_size_x / image_size_y
    

    #grid_size = [int(max(1.0, num_sqr * image_ratio) + 1), int(max(1.0, num_sqr / image_ratio) + 1)]

    
    full_step = [image_size_y/num_sqr, image_size_x/num_sqr]
    half_step = [full_step[0]/2.0, full_step[1]/2.0]
    
    matrix = [[[
        int(half_step[0] + x * full_step[0]),
            int(half_step[1] + y * full_step[1])
        ] for x in range(int(num_sqr))] for y in range(int(num_sqr))]
    centroids = list(chain.from_iterable(matrix))
    
    return centroids
    
def calculateDistance(pos1, c1, pos2, c2, num_of_pixels, num_of_clusters):
    """distance between two pixels"""
    s = sqrt(num_of_pixels/num_of_clusters)
    m = 5
    
    dist_norm = LA.norm(np.array(pos1) - np.array(pos2))
    color_norm = LA.norm(np.array(c1) - np.array(c2))
    
    return sqrt(((dist_norm**2)/s) + ((color_norm**2)/m))
    
def snic(image, compactness, num_of_clusters):
    print("Starting snic..")
    rows = im.shape[0]
    cols = im.shape[1]
    image_size = [rows,cols]
    
    num_of_pixels = rows*cols
    print("rows: ",rows)
    print("cols: ",cols)
    print("pixels: ",num_of_pixels)
    
    """ initialize centroids; C = [(x1,y1),...] ---> centroids = [(pos, avg color, num_pixels), ...]"""
    #C = initializeCentroid(image.shape, num_of_clusters)
    
    indices_rows = np.random.randint(0,rows,num_of_clusters)
    indices_cols = np.random.randint(0,cols,num_of_clusters)
    C = []
    for (i,j) in zip(indices_rows,indices_cols):
            C.append([i,j])
    
    print("number of centroids: ", len(C))
    #print("First centroid is (x,y): ", C[0][0], C[0][1])
    #print("Last centroid is (x,y): ", C[6][0], C[6][1])
    
    """ centroids = [[0,0,0]] initially there are 0 pixels"""
    centroids = [[pos, im[pos[0]][pos[1]], 0] for pos in C]
    
    """ initialize labels as 0"""
    labels = np.zeros((rows,cols), dtype = int)
    
    """create priority queue"""
    pq = []
    heapq.heapify(pq) 
    
    """push each centroid into pque"""
    for k in range(num_of_clusters):
        centroid_val = [C[k], image[C[k][0]][C[k][1]], k+1]
        e = element(0,centroid_val)
        heapq.heappush(pq,e) 
        
    """find label for each pixel"""
    while len(pq) != 0:
        curr_ele = heapq.heappop(pq)
        x = curr_ele.val[0]
        c = curr_ele.val[1]
        K = curr_ele.val[2]
        
        if labels[x[0]][x[1]] == 0:
            labels[x[0]][x[1]] = K
            
            """update centroid"""
            centroid = centroids[K-1]
            num = centroid[2]+1
            weight = 1/num
            centroid[0] = [
                           (centroid[0][0]*(1-weight)) + (x[0]*weight),
                           (centroid[0][1]*(1-weight)) + (x[1]*weight)
                           ]

            centroid[1] = [
                            (centroid[1][0] * (1 - weight)) + (c[0] * weight),
                            (centroid[1][1] * (1 - weight)) + (c[1] * weight),
                            (centroid[1][2] * (1 - weight)) + (c[2] * weight)]
            centroid[2] = num

            
            """retrieve neightbours' coordinates; nbh = [(x1,y1),...]"""
            nbh = neighbours(x, image_size)
            
            for n in nbh:
                pos = n
                color = image[pos[0]][pos[1]]
                
                value = [pos, color, K]
                key = calculateDistance(n, image[n[0]][n[1]], C[K-1], image[C[K-1][0]][C[K-1][1]], num_of_pixels, k)
                
                e = element(key, value)
                
                if labels[pos[0]][pos[1]] == 0:
                    heapq.heappush(pq,e)
            
    
    """output lables"""
    return labels, centroids
    
"""call snic()"""
image = Image.open('253036.jpg')
im = np.array(image)
#im = cv2.imread('C:/Users/muska/OneDrive/Desktop/Trimester 1/btp/orchid.jpg')
plt.imshow(im)
print(im.shape)

rows = im.shape[0]
#print("rows: ",rows)

cols = im.shape[1]
#print("cols: ",cols)

num_of_pixels = rows*cols

num_of_segments = 50
compactness = 0.05

labels, centroids = snic(im,compactness,num_of_segments)
#print(labels)

print("Shape of labels: ", labels.shape)
print("First 10 distinct labels are: ")
"""
for l in range(10):
    i = l*40 + 20
    print("label no. ", i, 0 ," is: ", labels[i][0])
"""
fig = plt.figure()
plt.imshow(mark_boundaries(im, labels, color=(1, 1, 1)))
plt.show()
bound = find_boundaries(labels, mode='thick').astype(np.uint8)


file = open("253036.seg","r")
gt = file.readlines()
print(gt[11])
print(len(gt))
gt_labels = np.zeros((rows,cols), dtype = int)

for i in range(11,len(gt)):
    temp = [int(i) for i in gt[i].split()]
    label_no =  temp[0]
    row_no = temp[1]
    col_s = temp[2]
    col_e = temp[3]
    for j in range(col_s,col_e+1):
        gt_labels[row_no][j]=label_no
    
    
fig = plt.figure()
plt.imshow(mark_boundaries(im, gt_labels, color=(1, 1, 1)))
plt.show()
gt_bound = find_boundaries(gt_labels, mode='thick').astype(np.uint8)

FP=0
TP=0
total = 0
for i in range(rows):
    for j in range(cols):
        gt_flag = gt_bound[i][j]
        total = total + gt_flag
        flag = 0
        arr1 = [-1,-1,-1, 0,0,0,1,1,1]
        arr2 = [-1, 0, 1, -1, 0, 1,-1,0,1]
        for (x,y) in zip(arr1,arr2):
            if (i+x<rows and i+x>=0 and j+y<cols and j+y>=0) and bound[i+x][j+y] == 1:
                flag = 1
                break
        TP = TP + gt_flag*flag
        FP = FP + (1-gt_flag*flag)
    
Recall = TP/total
print(Recall," ")
Precision = TP/(TP+FP)
print(Precision)



