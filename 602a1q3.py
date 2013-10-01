from __future__ import division

import numpy as np
import scipy as sp
from scipy.optimize import fminbound
import matplotlib.pyplot as plt

# Define initial parameters (loop over distribution of parameter options later)
alpha = 0.5
beta = 0.9
domain = np.linspace(1e-2, 10, 100)

def bellman(v):
    """
    Input: A function defined in terms of a discontinuous array of its output,
        where the domain is specifie above.
    Output: A transformation of the given function, using a contraction mapping
        defined by the Bellman equation at hand.
    """
    vInterpolate = lambda x: sp.interp(x, domain, v) # Use v as a function
    Tv = np.empty(domain.size)
    for i, j in enumerate(domain): # i is index, j is value at index
        objective = lambda c: - np.log(c) - beta*vInterpolate(j**alpha - c)
        c_star = fminbound(objective, 1e-2, j**alpha)
        Tv[i] = -objective(c_star)
    return Tv


v = np.ones(domain.size) # Initial guess
plt.figure(1)
for i in range(20):
    v = bellman(v)
    plt.plot(domain, v, color=plt.cm.autumn(i/20))

plt.show()