# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 14:27:28 2022

@author: kevin
"""

import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt

p = np.array([i/20 for i in range(1,401)])

j = np.linspace(-0, 3, 100)
k = np.linspace(-3, 0, 100)
def lp_circle(x, p, R=1):
    y = np.power((R**p - x**p), (1/p))
    return y

def sigmoid(x, c):
    return 1/(1 + np.exp(-x*c))

def sigmoid_algebraic(x, c):
    return 0.5 + (0.5*x)/np.power((1 + np.absolute(x)**c), 1/c)

areas = np.array([integrate.quad(lp_circle, 0, 1, args = q)[0] for q in p])
sig_pos = sigmoid(j, 1.86)
sig_neg = sigmoid_algebraic(k, 5)

plt.plot(np.log(p), areas)
# plt.xlabel('p')
# plt.ylabel('Area (R=1)')
# plt.show()
plt.plot(j, sig_pos)
plt.plot(k, sig_neg)

#%%

def exp_sigmoid_algebraic(x, c):
    return 0.5 +(0.5 * np.log(x))/np.power((1 + np.absolute(np.log(x))**c), 1/c)

def exp_sigmoid_tanh(x, c):
    return 0.5 + 0.5 * np.tanh(c* np.log(x))

def exp_sigmoid_algebraic_2(x):
    return (x)/np.power((1 + x**x), 1/x)

p_01 = np.array([i/30 for i in range(1, 31)])
area_01 = np.array([integrate.quad(lp_circle, 0, 1, args = q)[0] for q in p_01])

sig_exp_alg = exp_sigmoid_algebraic(p_01, 4.5)
sig_exp_tanh = exp_sigmoid_tanh(p_01, 1.5)
sig_exp_alg2 = exp_sigmoid_algebraic_2(p_01)

plt.plot(p_01, area_01, label = 'L-p space areas')
plt.plot(p_01, sig_exp_alg, label = 'modified sigmoid function x/(1+x**p)**(1/p)')
plt.plot(p_01, sig_exp_alg2)
# plt.plot(p_01, sig_exp_tanh)
plt.legend()
plt.xlim(0,1)
plt.xlabel('p')
plt.ylabel('Area (R=1)')
# plt.yscale('log')
plt.show()
