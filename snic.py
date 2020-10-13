import numpy as np
import heapq
import cv2

from numpy import linalg as LA
from math import sqrt
from itertools import chain

"""
initializeCentroid(Khobragade)
updateDistance(Khobragade)
neighbours
call snic
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
    
def initializeCentroid(image_size, image):
    """output list of centroids(pixel)"""
    """image_size = image.shape"""

    compactness = 10
    num_of_clusters = 100
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
    
def calculateDistance(curr_pixel, centroid_pixel):
    """distance between two pixels"""
    s = sqrt(num_of_pixels/num_of_clusters)
    m = 10
    dist_norm = LA.norm(curr_pixel.x - centroid_pixel.x)
    color_norm = LA.norm(curr_pixel.c - centroid_pixel.c)
    return math.sqrt(((dist_norm**2)/s) + ((color_norm**2)/m))

def updateDistance(curr_ele):
    
def snic(image,rows,cols,num_of_clusters):
    """ initialize centroids; C = [(x1,y1),...]"""
    C = initializeCentroid(image.shape, image)
    centroids = [[pos, im[pos[1]][pos[0]], 0] for pos in C]
    """ initialize labels as 0"""
    labels = np.zeros((rows,cols))
    
    """create priority queue"""
    pq = []
    heapq.heapify(pq) 
    
    """push each centroid into pque"""
    for k in range(num_of_clusters):
        e = element(C[k].x,C[k].c)
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
                
                e = element(pos,color)
                e.k = K
                e.d = calculateDistance(n,C[K],image)
                
                if labels[pos[0]][pos[1]] == 0:
                    heapq.heappush(pq,e)
            
    
    """output lables"""
    return labels
    
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
