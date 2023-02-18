import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpi4py import MPI
from datetime import *
from SSCalculation import *
from Tree import *
import math
from sklearn.metrics import r2_score
import time
np.random.seed(10)
clientNum = 4
class LeastSquareLoss:
    def gradient(self, actual, predicted):
        return -(actual - predicted)

    def hess(self, actual, predicted):
        return np.ones_like(actual)

class LogLoss():
    def gradient(self, actual, predicted):
        prob = 1.0 / (1.0 + np.exp(-predicted))
        return prob - actual

    def hess(self, actual, predicted):
        prob = 1.0 / (1.0 + np.exp(-predicted))
        return prob * (1.0 - prob) # Mind the dimension

class VerticalXGBoostClassifier:

    def __init__(self, rank, lossfunc, splitclass, _lambda=1, _gamma=0.5, _epsilon=0.1, n_estimators=3, max_depth=3):

        if lossfunc == 'LogLoss':
            self.loss = LogLoss()
        else:
            self.loss = LeastSquareLoss()
        self._lambda = _lambda
        self._gamma = _gamma
        self._epsilon = _epsilon
        self.n_estimators = n_estimators  # Number of trees
        self.max_depth = max_depth  # Maximum depth for tree
        self.rank = rank
        self.trees = []
        self.splitclass = splitclass
        for _ in range(n_estimators):
            tree = VerticalXGBoostTree(rank=self.rank,
                                       lossfunc=self.loss,
                                       splitclass=self.splitclass,
                                       _lambda=self._lambda,
                                        _gamma=self._gamma,
                                       _epsilon=self._epsilon,
                                       _maxdepth=self.max_depth,
                                       clientNum=clientNum)
            self.trees.append(tree)


    def getQuantile(self, colidx):
        # does some splitting thing needed for XGBoost I guess
        split_list = []
        if self.rank != 0: # For client nodes
            data = self.data.copy()
            idx = np.argsort(data[:, colidx], axis=0)
            data = data[idx]
            value_list = sorted(list(set(list(data[:, colidx]))))  # Record all the different value
            hess = np.ones_like(data[:, colidx])
            data = np.concatenate((data, hess.reshape(-1, 1)), axis=1)
            sum_hess = np.sum(hess)
            last = value_list[0]
            i = 1
            if len(value_list) == 1: # For those who has only one value, do such process.
                last_cursor = last
            else:
                last_cursor = value_list[1]
            split_list.append((-np.inf, value_list[0]))
            while i < len(value_list):
                cursor = value_list[i]
                small_hess = np.sum(data[:, -1][data[:, colidx] <= last]) / sum_hess
                big_hess = np.sum(data[:, -1][data[:, colidx] <= cursor]) / sum_hess
                if np.abs(big_hess - small_hess) < self._epsilon:
                    last_cursor = cursor
                else:
                    judge = value_list.index(cursor) - value_list.index(last)
                    if judge == 1: # Although it didn't satisfy the criterion, it has no more split, so we must add it.
                        split_list.append((last, cursor))
                        last = cursor
                    else: # Move forward and record the last.
                        split_list.append((last, last_cursor))
                        last = last_cursor
                        last_cursor = cursor
                i += 1
            if split_list[-1][1] != value_list[-1]:
                split_list.append((split_list[-1][1], value_list[-1]))  # Add the top value into split_list.
            split_list = np.array(split_list)
        return split_list

    def getAllQuantile(self):  # Global quantile, must be calculated before tree building, avoiding recursion.
        self_maxlen = 0
        if self.rank != 0:


            # for each input feature, it executes a getQuantile
            dict = {i:self.getQuantile(i) for i in range(self.data.shape[1])} # record all the split
            self_maxlen = max([len(dict[i]) for i in dict.keys()])
        else:

            dict = {}

        recv_maxlen = comm.gather(self_maxlen, root=1)
        maxlen = None
        if self.rank == 1:
            maxlen = max(recv_maxlen)

        # I think this essentially gets the maximum number of splits across all features in a distributed way
        self.maxSplitNum = comm.bcast(maxlen, root=1)
        # print('MaxSplitNum: ', self.maxSplitNum)
        self.quantile = dict

    def fit(self, X, y):
        data_num = X.shape[0]

        # reshape data to Nx1
        y = np.reshape(y, (data_num, 1))
        y_pred = np.zeros(np.shape(y))
        self.data = X.copy()
        self.getAllQuantile()  # some preprocessing to set split bounds
        for i in range(self.n_estimators):
            tree = self.trees[i]
            # get the data, maxsplit numbers and the dictionary of something to do with the splits (quantile). Perhaps the candidate values?
            tree.data, tree.maxSplitNum, tree.quantile = self.data, self.maxSplitNum, self.quantile
            y_and_pred = np.concatenate((y, y_pred), axis=1)

            """
            y_and_pred is a concatenation of 0s to the initial Y thats passed as labels (pseudo). 
            """
            tree.fit(y_and_pred, i)
            if i == self.n_estimators - 1: # The last tree, no need for prediction update.
                continue
            else:
                update_pred = tree.predict(X)
            if self.rank == 1:
                update_pred = np.reshape(update_pred, (data_num, 1))
                y_pred += update_pred

    def predict(self, X):
        y_pred = None
        data_num = X.shape[0]
        # Make predictions
        for tree in self.trees:
            # Estimate gradient and update prediction
            update_pred = tree.predict(X)
            if y_pred is None:
                y_pred = np.zeros_like(update_pred).reshape(data_num, -1)
            if self.rank == 1:
                update_pred = np.reshape(update_pred, (data_num, 1))
                y_pred += update_pred
        return y_pred

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def main1():
    data = pd.read_csv('./iris.csv').values

    zero_index = data[:, -1] == 0
    one_index = data[:, -1] == 1
    zero_data = data[zero_index]
    one_data = data[one_index]
    train_size_zero = int(zero_data.shape[0] * 0.8)
    train_size_one = int(one_data.shape[0] * 0.8)
    X_train, X_test = np.concatenate((zero_data[:train_size_zero, :-1], one_data[:train_size_one, :-1]), 0), \
                      np.concatenate((zero_data[train_size_zero:, :-1], one_data[train_size_one:, :-1]), 0)
    y_train, y_test = np.concatenate((zero_data[:train_size_zero, -1].reshape(-1,1), one_data[:train_size_one, -1].reshape(-1, 1)), 0), \
                      np.concatenate((zero_data[train_size_zero:, -1].reshape(-1, 1), one_data[train_size_one:, -1].reshape(-1, 1)), 0)

    X_train_A = X_train[:, 0].reshape(-1, 1)
    X_train_B = X_train[:, 2].reshape(-1, 1)
    X_train_C = X_train[:, 1].reshape(-1, 1)
    X_train_D = X_train[:, 3].reshape(-1, 1)
    X_test_A = X_test[:, 0].reshape(-1, 1)
    X_test_B = X_test[:, 2].reshape(-1, 1)
    X_test_C = X_test[:, 1].reshape(-1, 1)
    X_test_D = X_test[:, 3].reshape(-1, 1)
    splitclass = SSCalculate()
    model = VerticalXGBoostClassifier(rank=rank, lossfunc='LogLoss', splitclass=splitclass)

    if rank == 1:
        model.fit(X_train_A, y_train)
        print('end 1')
    elif rank == 2:
        model.fit(X_train_B, np.zeros_like(y_train))
        print('end 2')
    elif rank == 3:
        model.fit(X_train_C, np.zeros_like(y_train))
        print('end 3')
    elif rank == 4:
        model.fit(X_train_D, np.zeros_like(y_train))
        print('end 4')
    else:
        model.fit(np.zeros_like(X_train_B), np.zeros_like(y_train))
        print('end 0')

    if rank == 1:
        y_pred = model.predict(X_test_A)
    elif rank == 2:

        y_pred = model.predict(X_test_B)
    elif rank == 3:

        y_pred = model.predict(X_test_C)
    elif rank == 4:

        y_pred = model.predict(X_test_D)
    else:

        model.predict(np.zeros_like(X_test_A))

    if rank == 1:
        y_ori = y_pred.copy()
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
        result = y_pred - y_test
        print(np.sum(result == 0) / y_pred.shape[0])
        # for i in range(y_test.shape[0]):
        #     print(y_test[i], y_pred[i], y_ori[i])

def main2():
    data = pd.read_csv('./GiveMeSomeCredit/cs-training.csv')
    data.dropna(inplace=True)

    # ignore the sample ID column
    data = data[['SeriousDlqin2yrs',
       'RevolvingUtilizationOfUnsecuredLines', 'age',
       'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome',
       'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate',
       'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
       'NumberOfDependents']].values
    ori_data = data.copy()
    # Add features
    # for i in range(1):
    #     data = np.concatenate((data, ori_data[:, 1:]), axis=1)

    # normalizing data
    data = data / data.max(axis=0)

    ratio = 10000 / data.shape[0]

    # distinguish between samples where the first column is zero and one
    zero_index = data[:, 0] == 0
    one_index = data[:, 0] == 1
    zero_data = data[zero_index]
    one_data = data[one_index]
    zero_ratio = len(zero_data) / data.shape[0]
    one_ratio = len(one_data) / data.shape[0]
    num = 7500

    # ensure the number of samples from either class is the same in ratio as the whole dataset. They're training on 10000 samples here
    train_size_zero = int(zero_data.shape[0] * ratio) + 1
    train_size_one = int(one_data.shape[0] * ratio)
    X_train, X_test = np.concatenate((zero_data[:train_size_zero, 1:], one_data[:train_size_one, 1:]), 0), \
                      np.concatenate((zero_data[train_size_zero:train_size_zero+int(num * zero_ratio)+1, 1:], one_data[train_size_one:train_size_one+int(num * one_ratio), 1:]), 0)
    y_train, y_test = np.concatenate(
        (zero_data[:train_size_zero, 0].reshape(-1, 1), one_data[:train_size_one, 0].reshape(-1, 1)), 0), \
                      np.concatenate((zero_data[train_size_zero:train_size_zero+int(num * zero_ratio)+1, 0].reshape(-1, 1),
                                      one_data[train_size_one:train_size_one+int(num * one_ratio), 0].reshape(-1, 1)), 0)

    # partition the features
    X_train_A = X_train[:, :2]
    X_train_B = X_train[:, 2:4]
    X_train_C = X_train[:, 4:7]
    X_train_D = X_train[:, 7:]
    X_test_A = X_test[:, :2]
    X_test_B = X_test[:, 2:4]
    X_test_C = X_test[:, 4:7]
    X_test_D = X_test[:, 7:]

    splitclass = SSCalculate()

    # Pre-sets variables and hyper-parameters for the vertical training
    model = VerticalXGBoostClassifier(rank=rank, lossfunc='LogLoss', splitclass=splitclass, max_depth=3, n_estimators=3, _epsilon=0.1)
    start = datetime.now()
    if rank == 1:
        # model 1, i.e., rank 1 is assumed to own the labels
        model.fit(X_train_A, y_train)
        end = datetime.now()
        # print('In fitting 1: ', end - start)

        time = end - start
        for i in range(clientNum + 1):
            if i == 1:
                pass
            else:
                time += comm.recv(source=i)
        # print(time / (clientNum + 1))
        final_time = time / (clientNum + 1)
        # print('end 1')
        # print(final_time)
    elif rank == 2:

        model.fit(X_train_B, np.zeros_like(y_train))
        end = datetime.now()
        comm.send(end - start, dest=1)
        # print('In fitting 2: ', end - start)
        # print('end 2')
    elif rank == 3:

        model.fit(X_train_C, np.zeros_like(y_train))
        end = datetime.now()
        # print('In fitting 3: ', end - start)
        comm.send(end - start, dest=1)
        # print('end 3')
    elif rank == 4:

        model.fit(X_train_D, np.zeros_like(y_train))
        end = datetime.now()
        # print('In fitting 4: ', end - start)
        comm.send(end - start, dest=1)
        # print('end 4')
    else:

        model.fit(np.zeros_like(X_train_B), np.zeros_like(y_train))
        end = datetime.now()
        # print('In fitting 0: ', end - start)
        comm.send(end - start, dest=1)
        # print('end 0')

    if rank == 1:

        y_pred = model.predict(X_test_A)

    elif rank == 2:
        y_pred = model.predict(X_test_B)
    elif rank == 3:
        y_pred = model.predict(X_test_C)
    elif rank == 4:
        y_pred = model.predict(X_test_D)
    else:
        model.predict(np.zeros_like(X_test_A))

    if rank == 1:
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        y_pred2 = y_pred.copy()
        y_pred2[y_pred2 > 0.5] = 1
        y_pred2[y_pred2 <= 0.5] = 0
        y_pred2 = y_pred2.reshape(-1,1)
        y_test = y_test.reshape(-1,1)
        result = y_pred2 - y_test
        print(np.sum(result == 0) / y_pred.shape[0])
        # for i in range(y_test.shape[0]):
        #     print(y_test[i], y_pred[i])

def main3():
    data = np.load('./adult.npy')
    data = data / data.max(axis=0)

    ratio = 0.8

    zero_index = data[:, 0] == 0
    one_index = data[:, 0] == 1
    zero_data = data[zero_index]
    one_data = data[one_index]

    train_size_zero = int(zero_data.shape[0] * ratio) + 1
    train_size_one = int(one_data.shape[0] * ratio)

    X_train, X_test = np.concatenate((zero_data[:train_size_zero, 1:], one_data[:train_size_one, 1:]), 0), \
                      np.concatenate((zero_data[train_size_zero:, 1:], one_data[train_size_one:, 1:]), 0)
    y_train, y_test = np.concatenate(
        (zero_data[:train_size_zero, 0].reshape(-1, 1), one_data[:train_size_one, 0].reshape(-1, 1)), 0), \
                      np.concatenate((zero_data[train_size_zero:, 0].reshape(-1, 1),
                                      one_data[train_size_one:, 0].reshape(-1, 1)), 0)

    segment_A = int(0.2 * (data.shape[1] - 1))
    segment_B = segment_A + int(0.2 * (data.shape[1] - 1))
    segment_C = segment_B + int(0.3 * (data.shape[1] - 1))

    X_train_A = X_train[:, 0:segment_A]
    X_train_B = X_train[:, segment_A:segment_B]
    X_train_C = X_train[:, segment_B:segment_C]
    X_train_D = X_train[:, segment_C:]
    X_test_A = X_test[:, :segment_A]
    X_test_B = X_test[:, segment_A:segment_B]
    X_test_C = X_test[:, segment_B:segment_C]
    X_test_D = X_test[:, segment_C:]

    # Initialize something to do all the secret sharing operations
    splitclass = SSCalculate()
    model = VerticalXGBoostClassifier(rank=rank, lossfunc='LogLoss', splitclass=splitclass, max_depth=3, n_estimators=3, _epsilon=0.1)

    if rank == 1:
        start = datetime.now()
        model.fit(X_train_A, y_train)
        end = datetime.now()
        print('In fitting: ', end - start)
        print('end 1')
    elif rank == 2:
        model.fit(X_train_B, np.zeros_like(y_train))
        print('end 2')
    elif rank == 3:
        model.fit(X_train_C, np.zeros_like(y_train))
        print('end 3')
    elif rank == 4:
        model.fit(X_train_D, np.zeros_like(y_train))
    else:
        model.fit(np.zeros_like(X_train_B), np.zeros_like(y_train))
        print('end 0')

    if rank == 1:
        y_pred = model.predict(X_test_A)
    elif rank == 2:
        y_pred = model.predict(X_test_B)
    elif rank == 3:
        y_pred = model.predict(X_test_C)
    elif rank == 4:
        y_pred = model.predict(X_test_D)
    else:
        model.predict(np.zeros_like(X_test_A))

    if rank == 1:
        y_ori = y_pred.copy()
        y_pred = 1.0 / (1.0 + np.exp(-y_pred))
        y_pred2 = y_pred.copy()
        y_pred2[y_pred2 > 0.5] = 1
        y_pred2[y_pred2 <= 0.5] = 0
        y_pred2 = y_pred2.reshape(-1,1)
        y_test = y_test.reshape(-1,1)
        result = y_pred2 - y_test
        print(np.sum(result == 0) / y_pred.shape[0])
        # for i in range(y_test.shape[0]):
        #     print(y_test[i], y_pred[i])

def main4():
    data = pd.read_csv('AirQualityUCI/AirQualityUCI.csv',sep=';', decimal=',')
    data.drop(['Unnamed: 15','Unnamed: 16'],axis = 1,inplace = True)
    data.replace(to_replace=',', value='.', regex=True, inplace=True)

    for i in 'C6H6(GT) T RH AH'.split():
        data[i] = pd.to_numeric(data[i], errors='coerce')
    data.replace(to_replace=-200, value=np.nan, inplace=True)
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True).dt.date
    data['Time'] = pd.to_datetime(data['Time'], format='%H.%M.%S').dt.time

    data.drop('NMHC(GT)', axis=1, inplace=True)
    # data.dropna()
    data.drop_duplicates(inplace= True)
    data.dropna(how='any', axis=0, inplace=True)

    data.reset_index(drop=True,inplace=True)
    datetimecol = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str))
    data['DateTime'] = datetimecol
    data.drop(['Date', 'Time'],axis = 1,inplace = True)
    cols = data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data = data[cols]
    """ Drop the datetime column for now because it is of no use for this case """
    data.drop(['DateTime'], axis = 1, inplace = True)
    train_test_ratio = 0.6
    num_train_samples = int(train_test_ratio*len(data))

    # Let the y value be the prediction of carbon monoxide for now
    y_train, y_test = data.loc[:num_train_samples-1, ["CO(GT)"]], data.loc[num_train_samples:, ["CO(GT)"]]

    # drop the CO(GT) column from the dataset
    data.drop(["CO(GT)"], axis = 1, inplace = True)

    X_train, X_test = data.iloc[:num_train_samples, :], data.iloc[num_train_samples:, :]

    segment_A = int(0.2 * (data.shape[1] - 1))
    segment_B = segment_A + int(0.2 * (data.shape[1] - 1))
    segment_C = segment_B + int(0.3 * (data.shape[1] - 1))

    X_train_A = X_train.iloc[:, 0:segment_A].to_numpy()
    X_train_B = X_train.iloc[:, segment_A:segment_B].to_numpy()
    X_train_C = X_train.iloc[:, segment_B:segment_C].to_numpy()
    X_train_D = X_train.iloc[:, segment_C:].to_numpy()
    X_test_A = X_test.iloc[:, :segment_A].to_numpy()
    X_test_B = X_test.iloc[:, segment_A:segment_B].to_numpy()
    X_test_C = X_test.iloc[:, segment_B:segment_C].to_numpy()
    X_test_D = X_test.iloc[:, segment_C:].to_numpy()


    # Initialize something to do all the secret sharing operations
    splitclass = SSCalculate()
    model = VerticalXGBoostClassifier(rank=rank, lossfunc='mse', splitclass=splitclass, max_depth=3, n_estimators=3,
                                      _epsilon=0.1)

    start = datetime.now()
    if rank == 1:
        # model 1, i.e., rank 1 is assumed to own the labels

        model.fit(X_train_A, y_train)
        end = datetime.now()
        # print('In fitting 1: ', end - start)

        time = end - start
        for i in range(clientNum + 1):
            if i == 1:
                pass
            else:
                time += comm.recv(source=i)
        # print(time / (clientNum + 1))
        final_time = time / (clientNum + 1)
        # print('end 1')
        # print(final_time)
    elif rank == 2:

        model.fit(X_train_B, np.zeros_like(y_train))
        end = datetime.now()
        comm.send(end - start, dest=1)
        # print('In fitting 2: ', end - start)
        # print('end 2')
    elif rank == 3:

        model.fit(X_train_C, np.zeros_like(y_train))
        end = datetime.now()
        # print('In fitting 3: ', end - start)
        comm.send(end - start, dest=1)
        # print('end 3')
    elif rank == 4:

        model.fit(X_train_D, np.zeros_like(y_train))
        end = datetime.now()
        # print('In fitting 4: ', end - start)
        comm.send(end - start, dest=1)
        # print('end 4')
    else:
        temp1 = np.zeros_like(X_train_B)

        model.fit(np.zeros_like(X_train_B), np.zeros_like(y_train))
        end = datetime.now()
        # print('In fitting 0: ', end - start)
        comm.send(end - start, dest=1)
        # print('end 0')

    if rank == 1:
        y_pred_train = model.predict(X_train_A)
        y_pred = model.predict(X_test_A)

    elif rank == 2:
        y_pred_train = model.predict(X_train_B)
        y_pred = model.predict(X_test_B)
    elif rank == 3:
        y_pred_train = model.predict(X_train_C)
        y_pred = model.predict(X_test_C)
    elif rank == 4:
        y_pred_train = model.predict(X_train_D)
        y_pred = model.predict(X_test_D)
    else:
        model.predict(X_train_A)
        model.predict(np.zeros_like(X_test_A))

    if rank == 1:
        # plt.scatter(range(len(y_test)), y_test, color='blue')
        # plt.scatter(range(len(y_pred)), y_pred, color='red')
        # plt.show()
        print("r2 score train", r2_score(y_train, y_pred_train))
        print("r2 score test", r2_score(y_test, y_pred))

    print()

if __name__ == '__main__':
    # main2()
    main4()

