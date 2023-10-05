import numpy as np
import pandas as pd
from mpi4py import MPI
from SSCalculate_Alternative import *
from copy import deepcopy
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error as MSE

np.random.seed(10)
# clientNum = 8
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

bytes_sent = 0


def get_total_bytes(obj):
    size = sys.getsizeof(obj)
    if isinstance(obj, (list, tuple, set, dict)):
        size += sum(get_total_bytes(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        size += obj.nbytes
    elif hasattr(obj, '__dict__'):
        size += get_total_bytes(obj.__dict__)
    return size


# this is a splitter for scalar values
def CustomSSSplit(data, clientNum):
    r = np.array([np.random.uniform(0, 4) for i in range(clientNum - 1)])
    data = float(data)
    data -= np.sum(r, axis=0).astype('float64')
    data = np.expand_dims(data, axis=0)
    dataList = np.concatenate([r, data], axis=0)
    return dataList


# splits the value into shares and transmits
def SHR(val, source, splitclass, clientNum):
    global bytes_sent
    if rank == source:
        bytes_sent += sys.getsizeof(0)
        comm.send(val, 0)  # send it to the coordinator first
        return comm.recv(source=0)
    if rank == 0:
        val_r = comm.recv(source=source)
        if np.isscalar(val_r):
            shares = CustomSSSplit(val_r, clientNum)
        else:
            shares = splitclass.SSSplit(val_r, clientNum)
        for i in range(1, clientNum + 1):
            bytes_sent += get_total_bytes(shares[i - 1])
            comm.send(shares[i - 1], i)

        if np.isscalar(val_r):
            return 0.0
        else:
            return np.zeros(val_r.shape)
    else:
        return comm.recv(source=0)


# obtains the resulting value from the shares and stores this at the destination
def AGGSHR(val, dest):
    global bytes_sent
    data = comm.gather(val, root=dest)
    if rank == dest:
        data = data[1:]
        bytes_sent += get_total_bytes(data)
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

    if output_rows <= output_cols:
        for i in range(output_rows):
            A_vec = np.reshape(A[i, :], (-1, 1))
            A_mat = np.repeat(A_vec, repeats=output_cols, axis=1)
            result[i] = np.sum(splitclass.SMUL(A_mat, B, rank), axis=0)
    else:
        for i in range(output_cols):
            B_vec = np.reshape(B[:, i], (1, -1))
            B_mat = np.repeat(B_vec, repeats=output_rows, axis=0)
            result[:, i] = np.sum(splitclass.SMUL(A, B_mat, rank), axis=1)
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
def SMATINV(X, splitclass, clientNum):
    P = None
    if rank == clientNum:
        P = GenNonSingularMatrix(X.shape[0])
    P_orig = P
    P = SHR(P, clientNum, splitclass, clientNum)
    A_matrices = []

    for i in range(1, clientNum):
        tempMat = None
        if rank == i:
            tempMat = deepcopy(X)
        tempMat = SHR(tempMat, i, splitclass, clientNum)
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
    sumMats_P_inv = SHR(sumMats_P_inv, 1, splitclass, clientNum)
    local_res = SMATMUL(P, sumMats_P_inv, splitclass)
    return local_res


# Gets X^TX
def GramMatrix(X, splitclass):
    transpose_x = np.transpose(X)
    return SMATMUL(transpose_x, X, splitclass)


# Retrieves the optimal coefficients for a linear regression model using the normal equation
def NormalEqSolver(A, Y, splitclass, clientNum):
    gramA = GramMatrix(A, splitclass)  # A^T (A)
    local_res = SMATINV(gramA, splitclass, clientNum)
    A_trans = np.transpose(A)
    global_res = SMATMUL(local_res, A_trans, splitclass)
    final_res = SMATMUL(global_res, Y, splitclass)
    return final_res


if __name__ == "__main__":
    """Make regression cases of 10 features with 10 samples, 100 samples, and 1000 samples
    Let the number of clients vary from 2 to 5 to 10"""
    clientNum = comm.size - 1
    np.random.seed(123)
    methods = ["NE"]
    df = None
    try:
        df = pd.read_csv("results_comm.csv")
    except:
        pass
    X = Y = None
    alpha = 0.001
    features = [1000]
    samples = [1000]
    iters = [10]
    rows = []
    for sample in samples:
        for feat in features:
            if feat > sample:
                continue
            rowdata = {}
            rowdata["Features"] = feat
            rowdata["Samples"] = sample
            rowdata["Clients"] = clientNum
            if rank == 1:
                X, Y = make_regression(sample, feat, coef=False)
                minmax_scaler = MinMaxScaler()
                X = minmax_scaler.fit_transform(X)
                Y = np.reshape(Y, (Y.shape[0], 1))
                Y = minmax_scaler.fit_transform(Y)

            splitclass = SSCalculate(clientNum)
            X = SHR(X, 1, splitclass, clientNum)
            Y = SHR(Y, 1, splitclass, clientNum)
            for method in methods:
                if method == "NE":
                    optimal_coeffs = NormalEqSolver(X, Y, splitclass, clientNum)
                    bytes_sent += splitclass.bytes_sent
                    res = comm.gather(bytes_sent, 1)
                    if rank == 1:
                        rowdata["NE Bytes"] = np.sum(res)
                    bytes_sent = 0
                    splitclass.bytes_sent = 0
                elif method == "GD":
                    for iter in iters:
                        est_coeffs = np.random.random((X.shape[1], 1))
                        for i in range(iter):
                            prediction2 = SMATMUL(X, est_coeffs, splitclass)
                            transpose = np.transpose(X)
                            diff = Y - prediction2
                            gradient_distributed = 2 * alpha * SMATMUL(transpose, diff, splitclass)
                            est_coeffs = est_coeffs + gradient_distributed
                        bytes_sent += splitclass.bytes_sent
                        res = comm.gather(bytes_sent, 1)
                        if rank == 1:
                            rowdata["GD Bytes " + str(iter) + " iterations"] = np.sum(res)
                        bytes_sent = 0
                        splitclass.bytes_sent = 0

            rows.append(rowdata)
    if rank == 1:
        df1 = pd.DataFrame(rows)
        if df is None:
            df1.to_csv("results_comm.csv")
        else:
            df = df.append(df1)
            df.to_csv("results_comm.csv")
    # coeffs = np.reshape(coeffs, (coeffs.shape[0], 1))
    # """Normal equation way"""
    # if method == "NE":
    #     optimal_coeffs = NormalEqSolver(X, Y, splitclass, clientNum)
    #     preds = SMATMUL(X, optimal_coeffs, splitclass)
    #     final_preds = AGGSHR(preds, 1)
    #     true_vals = AGGSHR(Y, 1)
    #     bytes_sent += splitclass.bytes_sent
    #     res = comm.gather(bytes_sent, 1)
    #     if rank == 1:
    #         mse = MSE(true_vals, final_preds)
    #         rowdata = {"features:", num_features, "num_samples:", num_samples, "Clients": clientNum, "Method": "NE", "Train loss (MSE)": mse, "Bytes sent": np.sum(res)}
    #         df.append(rowdata)
    #
    #     bytes_sent = 0
    #     splitclass.bytes_sent = 0
    #
    # """GD way"""
    # if method == "GD":
    #     est_coeffs = np.random.random((X.shape[1], 1))
    #     for i in range(steps):
    #         prediction2 = SMATMUL(X, est_coeffs, splitclass)
    #         transpose = np.transpose(X)
    #         diff = Y - prediction2
    #         gradient_distributed = 2 * alpha * SMATMUL(transpose, diff, splitclass)
    #         est_coeffs = est_coeffs + gradient_distributed
    #
    #     preds = SMATMUL(X, est_coeffs, splitclass)
    #     final_preds = AGGSHR(preds, 1)
    #     true_vals = AGGSHR(Y, 1)
    #     bytes_sent += splitclass.bytes_sent
    #     res = comm.gather(bytes_sent, 1)
    #     if rank == 1:
    #         mse = MSE(true_vals, final_preds)
    #         print("Samples:", num_samples, "clients:", clientNum, "GD", "lr:", alpha, "iter:", steps, "Train loss(MSE):", mse, "bytes sent:",
    #               np.sum(res))
    #
    #     bytes_sent = 0
    #     splitclass.bytes_sent = 0

    MPI.Finalize()
