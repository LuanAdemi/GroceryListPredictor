import random
import numpy as np
import matplotlib.pyplot as plt

def f(p_x, p_0, a_x, a_prev):
    if a_x == a_prev:
        if a_x == 1:
            return (p_x * p_0)**0.75
        else:
            return 1-((1-p_x)*(1-p_0))**0.75
    else:
        if a_x == 1:
            return p_0**1.5
        else:
            return 1-(1-p_0)**1.5


def generate(probs, prods):
    n = 28
    pxs = np.zeros((n,len(prods)))
    data = np.zeros((n,len(prods)))
    for i, prob in enumerate(probs):
        data[0][i] = np.random.choice([0,1], p=[1-prob,prob])
        px = f(prob,prob,data[0][1],-1)
        for j in range(1,n):
            data[j][i] = np.random.choice([0,1], p=[1-px,px])
            pxs[j][i] = px
            px = f(px,prob,data[j][i], data[j-1][i])
    return data, pxs

prods = ["Banane","Apfel","Brot"]
probs = [0.5,0.05,0.9]
data, pxs = generate(probs,prods)
f, (ax1, ax2, ax3) = plt.subplots(3,1)
ax1.plot(data[:,0])
ax2.plot(data[:,1])
ax3.plot(data[:,2])
ax1.plot(pxs[:,0])
ax2.plot(pxs[:,1])
ax3.plot(pxs[:,2])


f.show()
   