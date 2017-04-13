from celery import Celery
import numpy as np
import math
import ToolBox
import pickle

app = Celery('tasks', broker='amqp://guest@localhost//', backend='amqp')

#app.control.add_consumer('IOQueue', reply=True,destination=['worker1@example.com'])
app.control.add_consumer('IOQueue', reply=True)
app.conf.task_routes = {'tasks.saveNeighbors': {'queue': 'IOQueue'}}

@app.task()
def findNeighborsTask(model,indexes):#TODO: move out of this module
    return ToolBox.findNeighbors(model,indexes)

@app.task()
def reducer(lists):
    results = []
    for l in lists:
        results += l
    return results

@app.task()
def angleHistTask(neighborsGraph):
    return ToolBox.angleHist(neighborsGraph)

@app.task()
def printResults(results):
    print(results)
    return

@app.task(queue='IOQueue')
def saveNeighbors(neighborsGraph):
    file = open('Celery_neighbors.pkl', 'w')
    pickle.dump(neighborsGraph,file)
    return neighborsGraph #allow pass through for next task