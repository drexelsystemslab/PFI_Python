from celery import Celery
import numpy as np
import math


app = Celery('tasks', broker='amqp://guest@localhost//', backend='amqp')


def mag(x):
    return math.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2)


def clean_acos(cos_angle):
    return math.acos(min(1, max(cos_angle, -1)))

@app.task
def findNeighbors(model,indexes):
    neighbors = []
    print("Start Thread")
    for index in indexes:
        data = model
        tri = data.points[index, :]
        axiswise1 = np.where(data == np.hstack((tri[0:3], tri[0:3], tri[0:3])), True, False)
        axiswise2 = np.where(data == np.hstack((tri[3:6], tri[3:6], tri[3:6])), True, False)
        axiswise3 = np.where(data == np.hstack((tri[6:9], tri[6:9], tri[6:9])), True, False)
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
                    np.dot(data.normals[index], data.normals[j]) / (mag(data.normals[index]) * mag(data.normals[j])))])

        neighbors.append([index,local_neighbors])
    print("End Thread")
    return neighbors