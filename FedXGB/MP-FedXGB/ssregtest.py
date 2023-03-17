import numpy as np
import pandas as pd
from mpi4py import MPI
from SSCalculation import *

np.random.seed(10)
clientNum = 4
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


if __name__ == "__main__":
    splitclass = SSCalculate()

    """Example for testing if multiplication works"""
    # val_1 = None
    # val_2 = None
    # if rank == 1:
    #     val_1 = np.ones((2, 2))
    # if rank == 2:
    #     val_2 = np.array([[1, 2], [3, 4]])
    # val_1 = SHR(val_1, 1)
    # val_2 = SHR(val_2, 2)
    #
    # res = splitclass.SMUL(val_1, val_2, rank)
    #
    # print("my rank is ", rank)
    # print("the local product i have is", res)
    # prod = AGGSHR(res, 1)
    # print("the aggregate share i have is", prod)

    """Example solving a system of linear equation in two variables without bias term a1X + b1Y = c1 and a2X + b2Y = c2
    Optimizing each persons solution individually using linalg.solve and then summing will not give the right answer"""
    # a1 = None
    # a2 = None
    # b1 = None
    # b2 = None
    # c1 = None
    # c2 = None
    # x = None
    # y = None
    #
    # # let a1 belong to rank 1, a2 belong to rank 2 b1, b2 to rank 3, c1, c2 to rank 4
    # if rank == 1:
    #     a1 = 2
    # if rank == 2:
    #     a2 = 1
    # if rank == 3:
    #     b1 = 3
    #     b2 = -1
    # if rank == 4:
    #     c1 = 5
    #     c2 = 0
    #
    # a1 = SHR(a1, 1)
    # a2 = SHR(a2, 2)
    # b1 = SHR(b1, 3)
    # b2 = SHR(b2, 3)
    # c1 = SHR(c1, 4)
    # c2 = SHR(c2, 4)
    #
    # a = np.array([[a1, b1], [a2, b2]])
    # b = np.array([c1, c2])
    #
    # soln = np.zeros((2,))
    # if rank != 0:
    #     soln = np.linalg.solve(a, b)
    #     print("my rank", rank)
    #     print("my solution", soln)
    #
    #
    # results = AGGSHR(soln, 1)
    # if rank == 1:
    #     print("my rank is", rank)
    #     print("my solution is", results)

    """Solving the linear equation as above using gradient descent. No secret sharing for now"""
    # a = np.array([2, 1]).reshape((-1, 1))
    # b = np.array([3, -1]).reshape((-1, 1))
    # c = np.array([5, 0]).reshape((-1, 1))
    # x = np.random.uniform(0, 5)  # initial solutions
    # y = np.random.uniform(0, 5)
    # c_pred = a*x + b*y
    # differences = c - c_pred
    # squared_differences = differences**2
    # loss = np.mean(squared_differences)/2
    # gradient_x_vector = -a*c + a*a*x + a*b*y
    # gradient_y_vector = -b*c + b*a*x + b*b*y
    # gradient_x = np.mean(gradient_x_vector)
    # gradient_y = np.mean(gradient_y_vector)
    # learning_rate = 0.0001
    # while abs(max(gradient_x, gradient_y) * learning_rate) > 0.000001:
    #     # print("haha")
    #     # exit()
    #     print(x, y)
    #     x = x - learning_rate * gradient_x
    #     y = y - gradient_y * learning_rate
    #     gradient_x_vector = -a * c + a * a * x + a * b * y
    #     gradient_y_vector = -b * c + b * a * x + b * b * y
    #     gradient_x = np.mean(gradient_x_vector)
    #     gradient_y = np.mean(gradient_y_vector)
    # print(x, y)

    """Now try the same as above with secret sharing"""
    """Assume a belong with 1, b with 2, c with 3"""
    a, b, c = None, None, None
    if rank == 1:
        a = np.array([2, 1]).reshape((-1, 1))
    if rank == 2:
        b = np.array([3, -1]).reshape((-1, 1))
    if rank == 3:
        c = np.array([5, 0]).reshape((-1, 1))
    a = SHR(a, 1)
    b = SHR(b, 2)
    c = SHR(c, 3)
    x = np.random.uniform(0, 5)  # initial solutions
    y = np.random.uniform(0, 5)
    x = comm.bcast(x, 0)
    y = comm.bcast(y, 0)
    tr_a = np.array([2, 1]).reshape((-1, 1))
    tr_b = np.array([3, -1]).reshape((-1, 1))
    tr_c = np.array([5, 0]).reshape((-1, 1))
    tr_gradient_x_vector = -tr_a * tr_c + tr_a * tr_a * x + tr_a * tr_b * y
    tr_gradient_y_vector = -tr_b * tr_c + tr_b * tr_a * x + tr_b * tr_b * y
    tr_gradient_x = np.mean(tr_gradient_x_vector)
    tr_gradient_y = np.mean(tr_gradient_y_vector)
    gradient_x_vector = -splitclass.SMUL(a, c, rank) + splitclass.SMUL(a, a, rank) * x + splitclass.SMUL(a, b, rank) * y
    gradient_y_vector = -splitclass.SMUL(b, c, rank) + splitclass.SMUL(b, a, rank) * x + splitclass.SMUL(b, b, rank) * y
    # # gradient_y_vector = -b*c + b*a*x + b*b*y
    gradient_x = np.mean(gradient_x_vector)
    gradient_y = np.mean(gradient_y_vector)
    actual_gradient_x = AGGSHR(gradient_x, 0)
    actual_gradient_y = AGGSHR(gradient_y, 0)
    actual_gradient_x = comm.bcast(actual_gradient_x, root=0)
    actual_gradient_y = comm.bcast(actual_gradient_y, root=0)
    # print("actual_gradient", actual_gradient_y)
    # print("true gradient y", tr_gradient_y)
    learning_rate = 0.0001
    # while abs(max(actual_gradient_x, actual_gradient_y) * learning_rate) > 0.000001:
    #     x = x - learning_rate * actual_gradient_x
    #     y = y - actual_gradient_y * learning_rate
    #     gradient_x_vector = -splitclass.SMUL(a, c, rank) + splitclass.SMUL(a, a, rank) * x + splitclass.SMUL(a, b,
    #                                                                                                          rank) * y
    #     gradient_y_vector = -splitclass.SMUL(b, c, rank) + splitclass.SMUL(b, a, rank) * x + splitclass.SMUL(b, b,
    #                                                                                                          rank) * y
    #     gradient_x = np.mean(gradient_x_vector)
    #     gradient_y = np.mean(gradient_y_vector)
    #     actual_gradient_x = AGGSHR(gradient_x, 0)
    #     actual_gradient_y = AGGSHR(gradient_y, 0)
    #     actual_gradient_x = comm.bcast(actual_gradient_x, root=0)
    #     actual_gradient_y = comm.bcast(actual_gradient_y, root=0)

    for i in range(40000):
        # if rank == 0:
        #     print(x)
        x = x - learning_rate * actual_gradient_x
        y = y - actual_gradient_y * learning_rate

        # tr_gradient_x_vector = -tr_a * tr_c + tr_a * tr_a * x + tr_a * tr_b * y
        # tr_gradient_x = np.mean(tr_gradient_x_vector)
        gradient_x_vector = -splitclass.SMUL(a, c, rank) + splitclass.SMUL(a, a, rank) * x + splitclass.SMUL(a, b,
                                                                                                             rank) * y
        gradient_y_vector = -splitclass.SMUL(b, c, rank) + splitclass.SMUL(b, a, rank) * x + splitclass.SMUL(b, b,
                                                                                                             rank) * y
        gradient_x = np.mean(gradient_x_vector)
        gradient_y = np.mean(gradient_y_vector)
        actual_gradient_x = AGGSHR(gradient_x, 0)
        actual_gradient_y = AGGSHR(gradient_y, 0)
        actual_gradient_x = comm.bcast(actual_gradient_x, root=0)
        actual_gradient_y = comm.bcast(actual_gradient_y, root=0)

    if rank == 0:
        print("x and y",x, y)