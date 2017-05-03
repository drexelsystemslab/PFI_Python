import numpy as np
import math
import time
import pickle
from stl import mesh
import matplotlib.pyplot as plt
import sys
import trimesh

def angleHist(neighborsGraph):
    angles = neighborsGraph[:,2]
    # plot with degree lables
    (hist, labels) = np.histogram(angles,bins=np.linspace(0,2*math.pi,20),density=True)
    labels = labels[1:20]*180/3.14 #strip out the first label (==0) and convert from degrees to radians
    print(hist.shape)
    descriptor = [] # append hist and lables such that [[labels[0],hist[0]]
    for bin in range(0,len(hist)):
        descriptor.append([labels[bin],hist[bin]])
    return {"angleHist":descriptor}

def findNeighbors(model):
    adjacents = model.face_adjacency#returns a list of pairs of faces that share an edge
    angles = trimesh.geometry.vector_angle(model.face_normals[adjacents])#calculate the angle between the normal vectors of each of the neighbors

    N = adjacents.shape
    neighbors = np.zeros((N[0], N[1] + 1))#increase size of neighbors array from (n,2) to (n,3)
    neighbors[:, :-1] = adjacents#add the adjacents to this new array
    neighbors[:, 2] = angles#add the angle such that for each set of neighbors, neighbors = (neighbor1,neighbor2,angle)
    return neighbors

