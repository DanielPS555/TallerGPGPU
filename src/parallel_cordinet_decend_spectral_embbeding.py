import numpy as np


import random
import numpy.linalg as la
import scipy
import copy
from multiprocessing import Pool

### ---- FUNCIONES AUX

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


### ----- Funcion de desenso por gradiente serial y sin actualizacion de a bloque

def coordinate_descent(A, d, X=None, tol=1e-5):
    ## Modificado con los errores

    n = A.shape[0]
    M = np.ones(n) - np.eye(n)
    if X is None:
        X = np.random.rand(n, d)
    else:
        X = X.copy()

    R = X.T @ X
    fold = -1

    errores = []

    iter = 0
    while (abs((fold - cost_function(A, X, M)) / fold) > tol):

        fold = cost_function(A, X, M)
        for i in range(n):
            k = X[i, :][np.newaxis]  # (fila i)^t . (fila i)
            R -= k.T @ k
            X[i, :] = solve_linear_system(R, (A[i, :] @ X).T, X[i, :])
            k = X[i, :][np.newaxis]
            R += k.T @ k
        err = abs((fold - cost_function(A, X, M)) / fold)
        errores = errores + [err]
        print("Iteracion: " + str(iter) + " | Error relativo: " + str(err) + " | norma R: " + str(la.norm(R)))
        iter += 1
    return (X, errores)



### ----- Funcion de desenso por gradiente serial y y actualizacion por bloque

def coordinate_descent_random_filas(A, d, batch_size, verbose=False, X=None, tol=1e-5):
    """
    Calcula el embbeding via coordinate descent, pero se calcula el numero de batch_size en cada iteracion

    Parameters
    ----------
    A = Matriz del grafo
    d = Dimencion del embbeding
    batch_size = numero de latent_posicion a actualizar en simultanio (tiene que ser un numero entero)
    verbose = (False por defecto) si es True muestra el proceso
    X = Valor inicial del embedding
    tol = Tolerencia relativa antes de terminar de iterar

    Returns
    -------
    """

    n = A.shape[0]
    M = np.ones(n) - np.eye(n)
    if X is None:
        X = np.random.rand(n, d)
    else:
        X = X.copy()

    fold = -1

    errores = []

    iter = 0
    while (abs((fold - cost_function(A, X, M)) / fold) > tol):
        fold = cost_function(A, X, M)
        corrs_pendientes = set(range(n))

        while len(corrs_pendientes) > 0:

            if len(corrs_pendientes) < batch_size:
                corrs = corrs_pendientes
            else:
                corrs = set(random.sample(list(corrs_pendientes), batch_size))
            corrs_pendientes = corrs_pendientes - corrs

            R_o = X.T @ X
            X_acc = copy.deepcopy(X)

            for i in corrs:
                R = copy.deepcopy(R_o)
                X_int = copy.deepcopy(X)
                k = X_int[i, :][np.newaxis]  # (fila i)^t . (fila i)
                R -= k.T @ k
                X_acc[i, :] = solve_linear_system(R, (A[i, :] @ X_int).T, X_int[i, :])
            X = X_acc

        err = cost_function(A, X, M)
        errores = errores + [err]
        if verbose:
            print("Iteracion: " + str(iter) + " | Error relativo: " + str(err) + " | norma R: " + str(
                np.linalg.norm(R_o)))
        iter += 1
    return (X, errores)


### ----- Version en parallelo

def calcularFila(args):
    """
    Funcion auxiliar donde se calcula cada fila.
    Parameters
    ----------
    args = (fila, X,R,A)

    Returns
    -------

    """

    global A_r

    filas = args[0]
    X = args[1]
    R = args[2]
    X_nueva = []
    for i in filas:
        k = X[i, :][np.newaxis]  # (fila i)^t . (fila i)
        R -= k.T @ k
        X_nueva.append(solve_linear_system(R, (A_r[:,i] @ X).T, X[i, :]))
        R += k.T @ k
    return X_nueva


def coordinate_descent_random_filas_parallel(A, d, batch_size, p, utilizar_fun_costo = False, num_iteraciones = 20,verbose=False, tol=1e-5, ):
    """
    Calcula el embbeding via coordinate descent, pero se calcula el numero de batch_size en cada iteracion en paralelo. Como la funcion de costo no esta implementada para funcionar en paralelo, entonces se permite desactivarla y fijar un numero de iteraciones.
    Parameters
    ----------
    A = Matriz del grafo
    d = Dimencion del embbeding
    batch_size = numero de latent_posicion a actualizar en simultanio (tiene que ser un numero entero)
    p = numero de cores
    utilizar_fun_costo = (False por defecto) indica si utilizar o no la funcion de costo, en caso de no usarla se basa en el numero de iteraciones
    num_iteraciones = (20 por Defecto) solo se utiliza en caso de no usar la funcion de costo,
    tol = tolerancia relatica al usar la funcion de costo

    Returns Embbeding X
    -------

    """
    np.random.seed(2024)
    n = A.shape[0]
    M = np.ones(n) - np.eye(n)
    X = np.random.rand(n, d)

    fold = -1

    errores = []

    iter = 0

    global A_r
    A_r = A

    with Pool(p) as pool:
        while (not utilizar_fun_costo and  iter < num_iteraciones): # or (utilizar_fun_costo and abs((fold - cost_function(A, X, M))/fold) >tol ):

            if utilizar_fun_costo:
                fold = cost_function(A, X, M)

            corrs_pendientes = set(range(n))

            while len(corrs_pendientes) > 0:
                if len(corrs_pendientes) < batch_size:
                    corrs_step = corrs_pendientes
                else:
                    corrs_step = set(random.sample(list(corrs_pendientes), batch_size))

                corrs_pendientes = corrs_pendientes - corrs_step

                corrs_por_thread = np.array_split(np.array(list(corrs_step)), p)

                R = X.T @ X


                X_r = copy.deepcopy(X)

                nueva_x = list(pool.map(calcularFila, [(corrs_por_thread[i], X, R) for i in range(p)]))

                for index_l, l in enumerate(corrs_por_thread):
                    for index_i, i in enumerate(l):
                        X[i, :] = nueva_x[index_l][index_i]

            errores = errores + [fold]
            print("Iteracion: " + str(iter) + " | Error absoluto: " + str(fold) + " | norma R: " + str(la.norm(R)))
            iter += 1

    return (X, errores)

