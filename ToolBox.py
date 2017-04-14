import numpy as np
import math
import time
import pickle
from stl import mesh
import matplotlib.pyplot as plt
import sys

def mag(x):
    return math.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)

def clean_acos(cos_angle):
    return math.acos(min(1, max(cos_angle, -1)))

def angleHist(neighborsGraph):
    #print(neighborsGraph)
    assert (len(neighborsGraph)>0)#you can't make a histogram of a model that doesn't have any facets (maybe don't need this any more?)
    angles = []
    for facet in neighborsGraph:
        #rint(facet)
        if(len(facet) > 1):#TODO: seems like when we only have one thread workong on the graph we get a different result from the merge than we should
            for neighbor in facet[1]:
                #print(neighbor)
                angles.append(neighbor[1])
        else:
             for neighbor in facet[0][1]:
                #print(neighbor)
                angles.append(neighbor[1])

    # plot with degree lables
    (hist, labels) = np.histogram(angles,bins=np.linspace(0,2*math.pi,20),density=True)
    labels = labels[1:20]*180/3.14 #strip out the first label (==0) and convert from degrees to radians
    print(hist.shape)
    descriptor = [] # append hist and lables such that [[labels[0],hist[0]]
    for bin in range(0,len(hist)):
        descriptor.append([labels[bin],hist[bin]])
    return ["angleHist",descriptor]

def findNeighbors(model,indexes):#TODO: move out of this module
    neighbors = []
    print("Start Thread")
    npPoints = np.array(model["points"])
    for index in indexes:
        data = model
        tri = npPoints[index, :]
        axiswise1 = np.where(npPoints == np.hstack((tri[0:3], tri[0:3], tri[0:3])), True, False)
        axiswise2 = np.where(npPoints == np.hstack((tri[3:6], tri[3:6], tri[3:6])), True, False)
        axiswise3 = np.where(npPoints == np.hstack((tri[6:9], tri[6:9], tri[6:9])), True, False)
        local_neighbors = []
        for j in range(0, axiswise1.shape[0]):
            count = 0
            if (np.all(axiswise1[j, 0:3]) or np.all(axiswise1[j, 3:6]) or np.all(axiswise1[j, 6:9])):
                count += 1
            if (np.all(axiswise2[j, 0:3]) or np.all(axiswise2[j, 3:6]) or np.all(axiswise2[j, 6:9])):
                count += 1
            if (np.all(axiswise3[j, 0:3]) or np.all(axiswise3[j, 3:6]) or np.all(axiswise3[j, 6:9])):
                count += 1
            if (count == 2):
                # nearest_neighbors.append(i)#this triangle shares exactly 2 vertex with the original triangle
                local_neighbors.append([j, clean_acos(
                    np.dot(data["normals"][index], data["normals"][j]) / (mag(data["normals"][index]) * mag(data["normals"][j])))])

        neighbors.append([[index,local_neighbors]])
    print("End Thread")
    return neighbors


def loadNeighborsGraph(fileURL):
    try:
        filepath = fileURL.rsplit('/',1)[0]
        name = fileURL.split('/')[-1]
        neighbors = pickle.load(open(filepath+'/'+name+'_neighbors.pkl','rb'))
        return neighbors
    except (OSError, IOError) as e:
        return []#don't have a neighbor graph so return an empty list of neighbors

def chunker(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

