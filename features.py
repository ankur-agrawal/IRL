import numpy as np
import math

def reward(x, y, theta, speed, weights):
    # f = np.empty([2,1], dtype = 'float128')
    # f[0,:] = feature0(x,y,theta, speed)
    features = [feature1, feature2]
    r = 0
    f = []
    for n in range(0, len(features)):
        f.append(features[n](x,y,theta, speed))
        r = r+weights[0,n]*f[n]
    # f = np.vstack((f,feature2(x,y,theta, speed)))
    # f4 = feature4(x,y,theta, speed)
    # f5 = feature5(x,y,theta, speed)
    # f6 = feature6(x,y,theta, speed)
    # r = np.dot(weights, f)
    # r = f
    return r, f
# def feature0(x, y, theta, speed):
#     return 1
def feature1(x, y, theta, speed):
    f1 = np.exp(-(x**2+y**2)/(0.1**2))
    return f1
def feature2(x,y, theta, speed):
    mean_x = -speed*np.cos(theta)
    mean_y = -speed*np.sin(theta)
    f2 = np.exp(-((x-mean_x)**2+(y-mean_y)**2)/0.1**2)
    # print f2
    return f2
