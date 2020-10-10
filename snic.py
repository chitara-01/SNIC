import numpy as np
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
    
def snic(image,rows,cols,num_of_clusters):
    """ initialize centroids; C = [(x1,y1),...]"""
    C = initializeCentroid(image_size, image)
    
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
        
        if label[x.X][x.Y]==0:
            label[x.X][x.Y] = K
            
            """update centroid"""
            updateDistance(curr_ele)
            
            """retrieve neightbours' coordinates; nbh = [(x1,y1),...]"""
            nbh = _neighbours(curr_ele)
            
            for n in nbh:
                pos = n
                color = image[pos[0]][pos[1]]
                
                e = element(pos,color)
                e.k = K
                e.d = calculateDistance(n,C[K],image)
                
                if label[pos[0]][pos[1]] == 0:
                    heapq.heappush(pq,e)
            
    
    """output lables"""
    return labels
     
"""call snic()"""
im = cv2.imread('C:/Users/muska/OneDrive/Desktop/Trimester 1/btp/orchid.jpg')
#print(im.shape)

k = 100

rows = im.shape[0]
#print("rows: ",rows)

cols = im.shape[1]
#print("cols: ",cols)

labels = snic(im,rows,cols,k)
