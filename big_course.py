# -*- coding: utf-8 -*-
"""
Created on Sun Oct  4 18:39:46 2020

@author: illusion
"""

import numpy as np

import matplotlib.pyplot as plt

def bin2flt(bin_seed):
    res = [int(str(i),2)*Q+U_min for i in bin_seed]
    return np.array(res)

def fitness(flt_seed):
    x = flt_seed
    f = x*np.sin(10*np.pi*x)+2.0
    return f

def roulette_select(bin_seed):
    seed_fit = fitness(bin2flt(bin_seed))
    fit_sum = sum(seed_fit)
    seed_pro = seed_fit/fit_sum
    next_bin_seed = []
    roulette_pro = np.array([])
    roulette_pro= np.append(roulette_pro,seed_pro[0])
    for i in range(1,np.size(bin_seed)):
        roulette_pro = np.append(roulette_pro,seed_pro[i]+roulette_pro[i-1])  

#   选取M个子代
    for i in range(0,np.size(bin_seed)):
        select_pro = np.random.random()
        index = np.where(roulette_pro > select_pro)
        index = index[0][0]
        next_bin_seed.append(bin_seed[index])
    return next_bin_seed

    
def crossover(bin_seed):
    group = np.arange(M_size)
    np.random.shuffle(group)
    list_seed = [['0']*(l) for _ in range(M_size)]
    bin_seed = [list(i) for i in bin_seed]
    for i in range(0,M_size):
        for j in range(0,np.size(bin_seed[i])-2):
            t = j+l+2-np.size(bin_seed[i])    
            list_seed[i][t] = bin_seed[i][j+2]
    for i in range(0,M_pair):
        if (np.random.random()>1-Pc):
            first = group[i]
            second = group[i+M_pair]
            pos = np.random.randint(l)
            for j in range(pos,l):
                temp = list_seed[first][j]
                list_seed[first][j] = list_seed[second][j]
                list_seed[second][j] = temp
    return list_seed
 
def mutation(list_seed):
    for i in range(0,np.size(np.size(list_seed))):
        for j in range(0,np.size(np.size(list_seed[i]))):
            if (np.random.random()>1-Pm):
                if list_seed[i][j]=='0':
                    list_seed[i][j]='1'
                else:
                    list_seed[i][j]='0'
    str_seed = [''.join(i) for i in list_seed]
    return str_seed
                    
U_min = -1
U_max = 2
l = 22
Q = (U_max - U_min)/(2**l-1)
Pc = 0.75
Pm = 0.05
M_size = 50
M_pair = int(M_size/2)

x = np.linspace(U_min,U_max,1000)
y = x*np.sin(10*np.pi*x)+2.0

seed = np.zeros(M_size,dtype=np.int32)
for i in range(0,M_size):
    seed[i] = np.random.randint(0,2**l)
bin_seed = [bin(i) for i in seed]

t=0
while t<200:
    bin_seed = roulette_select(bin_seed)
    bin_seed = crossover(bin_seed)
    bin_seed = mutation(bin_seed)
    t=t+1

xx=bin2flt(bin_seed)
yy=fitness(bin2flt(bin_seed))
index = np.where(yy==np.max(yy))
index = index[0][0]
print(xx[index])
print(np.max(yy))
plt.plot(x,y)