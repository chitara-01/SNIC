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
    m = 10
    
    dist_norm = LA.norm(np.array(pos1) - np.array(pos2))
    color_norm = LA.norm(np.array(c1) - np.array(c2))
    
    return sqrt(((dist_norm**2)/s) + ((color_norm**2)/m))
    
def snic(image, compactness, num_of_clusters):
    print("Starting snic..")
    rows = image.shape[0]
    cols = image.shape[1]
    image_size = [rows,cols]
    
    num_of_pixels = rows*cols
    print("rows: ",rows)
    print("cols: ",cols)
    print("pixels: ",num_of_pixels)
    
    """ initialize centroids; C = [(x1,y1),...] ---> centroids = [(pos, avg color, num_pixels), ...]"""
    C = initializeCentroid(image.shape, num_of_clusters)
    print("number of centroids: ", len(C))
    print("First centroid is (x,y): ", C[0][0], C[0][1])
    print("Last centroid is (x,y): ", C[99][0], C[99][1])
    
    """ centroids = [[0,0,0]] initially there are 0 pixels"""
    i_pos = np.zeros(2)
    c_pos = np.zeros(3)
    centroids = [[i_pos, c_pos, 0] for i in range(len(C))]
    
    """ initialize labels as 0"""
    labels = np.zeros((rows,cols))
    
    """create priority queue"""
    pq = []
    heapq.heapify(pq) 
    
    """push each centroid into pque"""
    for k in range(num_of_clusters):
        centroid_val = [C[k], image[C[k][0]][C[k][1]], k+1]
        #print("centroid pos", C[k][0], C[k][1])
        #print("centroid color", C[k][0], C[k][1])
        #print("centroid number", centroid_val[2])
        e = element(0,centroid_val)
        heapq.heappush(pq,e) 
        
    """find label for each pixel"""
    while len(pq) != 0:
        curr_ele = heapq.heappop(pq)
        x = curr_ele.val[0]
        c = curr_ele.val[1]
        K = curr_ele.val[2]
        print("cluster", K)
        if labels[x[0]][x[1]] == 0:
            labels[x[0]][x[1]] = K
            
            """update centroid"""
            centroid = centroids[K-1]
            num = centroid[2]+1
            #print("centroid[2]", centroid[2], K)
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
            centroids[K-1] = centroid
            #print(centroids[K-1])
            
            """retrieve neightbours' coordinates; nbh = [(x1,y1),...]"""
            nbh = neighbours(x, image_size)
            
            for n in nbh:
                pos = n
                color = image[pos[0]][pos[1]]
                
                value = [pos, color, k]
                key = calculateDistance(n, image[n[0]][n[1]], C[K-1], image[C[K-1][0]][C[K-1][1]], num_of_pixels, k)
                
                e = element(key, value)
                
                if labels[pos[0]][pos[1]] == 0:
                    heapq.heappush(pq,e)
            
    
    """output lables"""
    return labels, centroids
    
"""call snic()"""
im= cv2.imread('C:/Users/muska/OneDrive/Desktop/Trimester 1/btp/orchid.jpg')
plt.imshow(im)
print(im.shape)

rows = im.shape[0]
#print("rows: ",rows)

cols = im.shape[1]
#print("cols: ",cols)

num_of_pixels = rows*cols

num_of_segments = 100
compactness = 10

labels, centroids = snic(im,compactness,num_of_segments)

print("Shape of labels: ", labels.shape)

print("First 10 distinct labels are: ")
for l in range(10):
    i = l*40 + 20
    print("label no. ", i, 0 ," is: ", labels[i][0])

fig = plt.figure("Segmented output with %d segments " % len(centroids))
plt.imshow(mark_boundaries(im, labels, color=(1, 1, 1)))
plt.show()
