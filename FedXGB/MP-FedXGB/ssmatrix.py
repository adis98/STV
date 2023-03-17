import numpy as np
import pandas as pd
from mpi4py import MPI
from SSCalculate_Alternative import *
from copy import deepcopy

np.random.seed(10)
clientNum = 2
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# this is a splitter for scalar values
def CustomSSSplit(data, clientNum):
    r = np.array([np.random.uniform(0, 4) for i in range(clientNum - 1)])
    data = float(data)
    data -= np.sum(r, axis=0).astype('float64')
    data = np.expand_dims(data, axis=0)
    dataList = np.concatenate([r, data], axis=0)
    return dataList


# splits the value into shares and transmits
def SHR(val, source, splitclass):
    if rank == source:
        comm.send(val, 0)  # send it to the coordinator first
        return comm.recv(source=0)
    if rank == 0:
        val_r = comm.recv(source=source)
        if np.isscalar(val_r):
            shares = CustomSSSplit(val_r, clientNum)
        else:
            shares = splitclass.SSSplit(val_r, clientNum)
        for i in range(1, clientNum + 1):
            comm.send(shares[i - 1], i)
        if np.isscalar(val_r):
            return 0.0
        else:
            return np.zeros(val_r.shape)
    else:
        return comm.recv(source=0)


# obtains the resulting value from the shares and stores this at the destination
def AGGSHR(val, dest):
    data = comm.gather(val, root=dest)
    if rank == dest:
        data = data[1:]
        return np.sum(data, axis=0)
    else:
        return val


# Secure Scalar product (dot product)
def SDOT(A, B, splitclass):
    return np.sum(splitclass.SMUL(A, B, rank))


def SMATMUL(A, B, splitclass):
    output_rows = A.shape[0]
    output_cols = B.shape[1]
    result = np.zeros((output_rows, output_cols))
    for i in range(output_rows):
        for j in range(output_cols):
            A_vec = np.reshape(A[i, :], (-1, 1))
            B_vec = np.reshape(B[:, j], (-1, 1))
            result[i, j] = SDOT(A_vec, B_vec, splitclass)
    return result


# Creates a non-singular matrix for shape dxd
def GenNonSingularMatrix(d):
    singular = True
    while singular:
        k = np.random.rand(d, d)
        if np.linalg.matrix_rank(k) == d:
            singular = False
    return k


# Computes the matrix inverse of a matrix X that's shared between two parties
def SMATINV(X, splitclass):
    P = None
    if rank == clientNum:
        P = GenNonSingularMatrix(X.shape[0])
    P_orig = P
    P = SHR(P, clientNum, splitclass)
    A_matrices = []

    for i in range(1, clientNum):
        tempMat = None
        if rank == i:
            tempMat = deepcopy(X)
        tempMat = SHR(tempMat, i, splitclass)
        A_matrices.append(tempMat)
    products = []
    for i in range(len(A_matrices)):
        currentA = A_matrices[i]
        products.append(SMATMUL(currentA, P, splitclass))

    sumProducts = np.sum(products, axis=0)

    if rank == clientNum:
        sumProducts = sumProducts + np.matmul(X, P_orig)

    sumMats_P = AGGSHR(sumProducts, 1)  # assume first client collects this
    sumMats_P_inv = None
    if rank == 1:
        # Computes the inverse
        sumMats_P_inv = np.linalg.pinv(sumMats_P)
    sumMats_P_inv = SHR(sumMats_P_inv, 1, splitclass)
    local_res = SMATMUL(P, sumMats_P_inv, splitclass)
    return local_res


# Gets X^TX
def GramMatrix(X):
    transpose_x = np.transpose(X)
    return SMATMUL(transpose_x, X, splitclass)


# Retrieves the optimal coefficients for a linear regression model using the normal equation
def NormalEqSolver(A, Y):
    gramA = GramMatrix(A)  # A^T (A)
    local_res = SMATINV(gramA, splitclass)
    A_trans = np.transpose(A)
    global_res = SMATMUL(local_res, A_trans, splitclass)
    final_res = SMATMUL(global_res, Y, splitclass)
    return final_res


if __name__ == "__main__":
    splitclass = SSCalculate(clientNum)
    A = None
    Y = None
    # The true matrix is [1,2,3,4]
    if rank == 1:
        A = np.array([[1, 2], [0, 0]])
        Y = np.array([[2], [1]])

    if rank == 2:
        A = np.array([[0, 0], [3, 4]])
        Y = np.zeros((2, 1))

    if rank == 0:
        A = np.zeros((2, 2))
        Y = np.zeros((2, 1))

    gramA = GramMatrix(A)  # A^T (A)

    optimal_coeffs = NormalEqSolver(A, Y)

    res = SMATMUL(A, optimal_coeffs, splitclass)

    res = AGGSHR(res, 1)
    if rank == 1:
        print(res)


