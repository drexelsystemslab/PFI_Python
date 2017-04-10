import numpy as np
import math
import time
import pickle
import tasks
from stl import mesh
import matplotlib.pyplot as plt

class ToolBox(object):
    def __init__(self,name,showPlot):
        self.name = name
        self.model = mesh.Mesh.from_file(name + '.stl')
        self.descriptor = []#initialize the descriptor vector
        self.showPlot = showPlot

    def getNeighborsGraph(self):
        try:
            neighbors = pickle.load(open(self.name+'_neighbors.pkl','rb'))
        except (OSError, IOError) as e:
            neighbors = self.generateGraph()
        return neighbors

    def generateGraph(self):
        start = time.time()
        seq = range(0,self.model.points.shape[0])
        num = 4 #num of chunks
        avg = len(seq) / float(num)
        out = []
        last = 0.0

        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg

        resultsHolder = [tasks.findNeighbors.delay(self.model,chunk) for chunk in out]

        while(all(x.ready() == True for x in resultsHolder)==False):
            print(time.time() - start);
            time.sleep(0.5);

        results = []
        for result in resultsHolder:
            results = results + result.get()[0]
        file = open(self.name+'_neighbors.pkl','w')
        pickle.dump(results,file)
        end = time.time()
        print(end - start)
        return results

    def angleHist(self):
        neighborsGraph = self.getNeighborsGraph()
        #print(neighborsGraph)
        angles = []
        for facet in neighborsGraph:
            #print(facet)
            for neighbor in facet:
                #print(neighbor)
                angles.append(neighbor[1])

        # plot with degree lables
        (hist, labels) = np.histogram(angles,bins=np.linspace(0,2*math.pi,20),density=True)
        labels = labels[1:20]*180/3.14 #strip out the first label (==0) and convert from degrees to radians
        print(hist.shape)
        descriptor = [] # append hist and lables such that [[labels[0],hist[0]]
        for bin in range(0,len(hist)):
            descriptor.append([labels[bin],hist[bin]])

        self.descriptor.append(["angleHist",descriptor])

        if(self.showPlot):
            plt.figure(1)
            plt.clf()
            plt.bar(labels,hist,width=10)
            plt.show()
        return