import numpy as np
import math
from numpy import linalg as LA
"""element
pixel
initializeCentroid
calculateDistance
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
        
def initializeCentroid(image_size, image):
    """output list of centroids(pixel)"""
    
def calculateDistance(curr_pixel, centroid_pixel):
    """distance between two pixels"""
    s = math.sqrt(num_of_pixels/num_of_clusters)
    m = 10
    dist_norm = LA.norm(curr_pixel.x - centroid_pixel.x)
    color_norm = LA.norm(curr_pixel.c - centroid_pixel.c)
    return math.sqrt(((dist_norm**2)/s) + ((color_norm**2)/m))
    
def snic(image,rows,cols,num_of_clusters):
    """ initialize centroids"""
    C = initializeCentroid(image_size, image)
    
    """ initialize labels as 0"""
    labels = (rows,cols)
    np.zeros(labels)
    
    """create priority queue"""
    pq = createPq()
    
    """push each centroid into pque"""
    for k in range(num_of_clusters):
        e = element(C[k].x,C[k].c)
        pq.push(e)
        
    """find label for each pixel"""
    while pq.empty()==False:
        curr_ele = pq.pop()
        x = curr_ele.x
        K = curr_ele.k
        
        if label[x.X][x.Y]==0:
            label[x.X][x.Y] = K
            
            """update centroid"""
            updateDistance(curr_ele)
            
            """retrieve neightbours"""
            nbh,num = _neighbours(curr_ele)
            
            for i in range(num):
                n = nbh[i]
                e = element(n.x,n.c)
                e.k = K
                e.d = calculateDistance(n,C[K])
                
                if label[n.X][n.Y]==0:
                    pq.push(e)
            
    
    """output lables"""
    return labels
    
"""call snic()"""
k = 100
labels = snic(image,rows,cols,k)
