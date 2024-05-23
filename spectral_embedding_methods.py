#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.linalg as la
import scipy
#Install the following fork of pymanopt in order to use the manifold of matrices with orthogonal columns
#pip install git+https://github.com/marfiori/pymanopt
#from pymanopt.manifolds import Stiefel_tilde

def coordinate_descent(A,d,X=None,tol=1e-5):
    """
    Solves the problem  min ||(A-XX^T)*M||_F^2
    by block coordinate descent.
    Here * is the entry-wise product.
    M is the matrix with zeros in the diagonal, and ones off-diagonal.    
    Returns X, solution of min ||(A-XX^T)*M||_F^2
    
    Parameters
    ----------
    A : matrix nxn
    d : dimension of the embedding
    X : initialization
    tol: tolerance used in the stop criterion  
        
    Returns
    -------
    Matrix X
        solution of the embedding problem
    """

    n=A.shape[0]
    M = np.ones(n) - np.eye(n)
    if X is None:
        X = np.random.rand(n,d)
    else:
        X = X.copy()
    
    R = X.T@X
    fold = -1
    while (abs((fold - cost_function(A, X, M))/fold) >tol):
        fold = cost_function(A, X, M)
        for i in range(n):
            k=X[i,:][np.newaxis]
            R -= k.T@k
            X[i,:] = solve_linear_system(R,(A[i,:]@X).T,X[i,:])
            k=X[i,:][np.newaxis]
            R += k.T@k

    return X


##### Auxiliary functions #####

def cost_function(A,X,M):
    """
    RDPG cost function ||(A-XX^T)*M||_F^2
    where * is the entry-wise product.

    Parameters
    ----------
    A : matrix nxn
    X : matrix of embeddings
    M : mask matrix nxn
        
    Returns
    -------
    value of ||(A-XX^T)*M||_F^2
    """
    return 0.5*np.linalg.norm((A - X@X.T)*M,ord='fro')**2

def solve_linear_system(A,b,xx):
    """
    Linear system solver, used in several methods.
    Should you use another method for solving linear systems, just change this function.
    
    Returns the solution of Ax=b
    Parameters
    ----------
    A : matrix nxn
    b : vector 

    Returns
    -------
    vector x
        solution to Ax=b

    """
    try:
        result = scipy.linalg.solve(A,b)
    except:
        result = scipy.sparse.linalg.minres(A,b,xx)[0]    
    return result
