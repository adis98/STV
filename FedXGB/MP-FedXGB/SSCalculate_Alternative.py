import numpy as np
import pandas as pd
from mpi4py import MPI
from datetime import *
import math
import time
from VerticalXGBoost import *
from Tree import *

np.random.seed(10)
import sys

comm = MPI.COMM_WORLD


def get_total_bytes(obj):
    size = sys.getsizeof(obj)
    if isinstance(obj, (list, tuple, set, dict)):
        size += sum(get_total_bytes(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        size += obj.nbytes
    elif hasattr(obj, '__dict__'):
        size += get_total_bytes(obj.__dict__)
    return size



class SSCalculate:

    def __init__(self, clientNum):
        self.clientNum = clientNum
        self.bytes_sent = 0

    # Partitions a list of number(s) among participants in a secret sharing way. i.e., they all sum up to the true value
    def SSSplit(self, data, clientNum):
        r = np.array([np.random.uniform(0, 4, (data.shape[0], data.shape[1])) for i in range(clientNum - 1)])
        data = data.astype('float64')
        data -= np.sum(r, axis=0).astype('float64')
        data = np.expand_dims(data, axis=0)
        dataList = np.concatenate([r, data], axis=0)
        return dataList

    def SMUL(self, data_A, data_B, rank):
        if len(data_A.shape) <= 1:
            data_A = data_A.reshape(-1, 1)
            data_B = data_B.reshape(-1, 1)
        if rank == 0:  # Send shared data
            a = np.random.rand(data_A.shape[0], data_A.shape[1])
            b = np.random.rand(data_A.shape[0], data_A.shape[1])
            c = a * b
            dataList_a = self.SSSplit(a, self.clientNum)
            dataList_b = self.SSSplit(b, self.clientNum)
            dataList_c = self.SSSplit(c, self.clientNum)
            for i in range(1, self.clientNum + 1):
                self.bytes_sent += get_total_bytes([dataList_a[i - 1], dataList_b[i - 1], dataList_c[i - 1]])
                comm.send([dataList_a[i - 1], dataList_b[i - 1], dataList_c[i - 1]], dest=i)
            # return a
            return np.zeros(data_A.shape)
        elif rank == 1:
            ra, rb, rc = comm.recv(source=0)
            ei = data_A - ra
            fi = data_B - rb
            eList = []
            fList = []
            for i in range(2, self.clientNum + 1):
                temp_e, temp_f = comm.recv(source=i)
                eList.append(temp_e)
                fList.append(temp_f)

            e = np.sum(np.array(eList), axis=0) + ei
            f = np.sum(np.array(fList), axis=0) + fi
            for i in range(2, self.clientNum + 1):
                self.bytes_sent += sys.getsizeof((e, f))
                comm.send((e, f), dest=i)
            zi = e * f + f * ra + e * rb + rc
            return zi
        else:
            ra, rb, rc = comm.recv(source=0)
            ei = data_A - ra
            fi = data_B - rb
            self.bytes_sent += sys.getsizeof((ei, fi))
            comm.send((ei, fi), dest=1)
            e, f = comm.recv(source=1)
            zi = f * ra + e * rb + rc
            return zi
