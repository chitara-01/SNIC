import numpy as np
import heapq
import cv2

import matplotlib.pyplot as plt
from numpy import linalg as LA
from math import sqrt
from itertools import chain
from skimage.segmentation import mark_boundaries

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

    if x+1 < image_size[1]:
        nbh.append([x+1, y])

    if y+1 < image_size[0]:
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
    
    full_step = [image_size_x/num_sqr, image_size_y/num_sqr]
    half_step = [full_step[0]/2.0, full_step[1]/2.0]
    
    matrix = [[[
        int(half_step[0] + x * full_step[0]),
            int(half_step[1] + y * full_step[1])
        ] for x in range(int(num_sqr))] for y in range(int(num_sqr))]
    centroids = list(chain.from_iterable(matrix))
    
    return centroids
    
def calculateDistance(curr_pixel, centroid_pixel, num_of_pixels, num_of_clusters):
    """distance between two pixels"""
    s = sqrt(num_of_pixels/num_of_clusters)
    m = 10
    dist_norm = LA.norm(curr_pixel.x - centroid_pixel.x)
    color_norm = LA.norm(curr_pixel.c - centroid_pixel.c)
    return sqrt(((dist_norm**2)/s) + ((color_norm**2)/m))
    
def snic(image, compactness, num_of_clusters):
    rows = im.shape[0]
    cols = im.shape[1]
    
    num_of_pixels = rows*cols
    
    """ initialize centroids; C = [(x1,y1),...] ---> centroids = [(pos, avg color, num_pixels), ...]"""
    C = initializeCentroid(image.shape, num_of_clusters)
    
    """ centroids = [[0,0,0]] initially there are 0 pixels"""
    centroids = [[pos, im[pos[1]][pos[0]], 0] for pos in C]
    
    """ initialize labels as 0"""
    labels = np.zeros((rows,cols))
    
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
        x = curr_ele.x
        K = curr_ele.k
        
        if labels[x.X][x.Y]==0:
            labels[x.X][x.Y] = K
            
            """update centroid"""
            centroid = centroids[K]
            num = centroid[2]+1
            weight = 1/num
            centroid[0] = [
                           (centroid[0][0]*(1-weight)) + (x[0]*weight),
                           (centroid[0][1]*(1-weight)) + (x[1]*weight)
                           ]

            centroid[1] = [
                            (centroid[1][0] * (1 - weight)) + (curr_ele.c[0] * weight),
                            (centroid[1][1] * (1 - weight)) + (curr_ele.c[1] * weight),
                            (centroid[1][2] * (1 - weight)) + (curr_ele.c[2] * weight)]
            centroid[2] = num

            
            """retrieve neightbours' coordinates; nbh = [(x1,y1),...]"""
            nbh = neighbours(curr_ele)
            
            for n in nbh:
                pos = n
                color = image[pos[0]][pos[1]]
                
                value = [pos, color, k]
                key = calculateDistance(n,C[K],rows*cols,k)
                
                e = element(key, value)
                
                if labels[pos[0]][pos[1]] == 0:
                    heapq.heappush(pq,e)
            
    
    """output lables"""
    return labels, centroids
    
"""call snic()"""
im = cv2.imread('C:/Users/muska/OneDrive/Desktop/Trimester 1/btp/orchid.jpg')
#plt.imshow(im)
print(im.shape)

rows = im.shape[0]
#print("rows: ",rows)

cols = im.shape[1]
#print("cols: ",cols)

num_of_pixels = rows*cols

num_of_segments = 100
compactness = 0.01

labels, centroids = snic(im,compactness,num_of_segments)

fig = plt.figure("Segmented output with %d segments " % len(centroids))
plt.imshow(mark_boundaries(im, np.array(labels), color=(1, 1, 1)))
plt.show()
