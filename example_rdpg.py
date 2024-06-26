#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy
import timeit
import networkx as nx
import importlib.util
import spectral_embedding_methods as sem

gy=0
if importlib.util.find_spec('graspologic') is None:
    print("graspologic is not installed. Running the other methods.")
else:
    gy=1
    import graspologic as gy

#%%
def embed_scipy(A, K):
    UA, SA, VAt = scipy.linalg.svd(A)
    XA = UA[:,0:K].dot(np.diag(np.sqrt(SA[0:K])))    
    return XA

p1 = [[0.5, 0.2],
      [0.2, 0.5]]

d=60

    
n = [1800, 1200]
nt=np.sum(n)
g = nx.stochastic_block_model(n,p1)
A = nx.to_numpy_array(g)
M = np.ones(nt) - np.eye(nt)

print("Running Coordinate descent")
start_cd = timeit.default_timer()
X_cd = sem.coordinate_descent(A,d)
stop_cd = timeit.default_timer()
time_cd = round(stop_cd-start_cd,3)
print("Done in ",time_cd," seconds.\n")

print("Running Scipy SVD")
start_scipy = timeit.default_timer()
X_scipy = embed_scipy(A,d)
stop_scipy = timeit.default_timer()
time_scipy = round(stop_scipy-start_scipy,3)
print("Done in ",time_scipy," seconds.\n")

if gy:
    print("Running Gaspologic ASE")
    ase = gy.embed.AdjacencySpectralEmbed(n_components=d, diag_aug=True, algorithm='full')
    start_gy = timeit.default_timer()
    X_gy = ase.fit_transform(A)
    stop_gy = timeit.default_timer()
    time_gy = round(stop_gy-start_gy,3)
    print("Done in ",time_gy," seconds.\n")

#%%
print("Method                        Time (s) \t Cost function ||(A-XX^T)*M||_F^2",)
print("---------------------------------------------------------------------",)
print("Coordinate descent            ",time_cd,"\t\t", sem.cost_function(A,X_cd,M))
print("Scipy                         ",time_scipy,"\t\t", sem.cost_function(A,X_scipy,M))
if gy: print("Graspologic                   ",time_gy,"\t\t", sem.cost_function(A,X_gy,M))
