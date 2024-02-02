import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from mpi4py import MPI
from pmdarima.arima import auto_arima
from SarimaPolynomial import SARIMAX_POLYPROCESSOR_VFL
from ssmatrix import AGGSHR
from matplotlib import pyplot as plt
import seaborn as sns

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
n_cli = comm.size - 1  # coordinator, rank 0 is not counted


def preprocess():
    np.random.seed(123)
    data = pd.read_csv('AirQualityUCI/AirQualityUCI.csv', sep=';', decimal=',')
    data.drop(['Unnamed: 15', 'Unnamed: 16'], axis=1, inplace=True)
    data.replace(to_replace=',', value='.', regex=True, inplace=True)

    for i in 'C6H6(GT) T RH AH'.split():
        data[i] = pd.to_numeric(data[i], errors='coerce')
    data.replace(to_replace=-200, value=np.nan, inplace=True)
    data['Date'] = pd.to_datetime(data['Date'], dayfirst=True).dt.date
    data['Time'] = pd.to_datetime(data['Time'], format='%H.%M.%S').dt.time
    data.drop('NMHC(GT)', axis=1, inplace=True)
    data.drop_duplicates(inplace=True)
    data.dropna(how='any', axis=0, inplace=True)

    data.reset_index(drop=True, inplace=True)
    datetimecol = pd.to_datetime(data['Date'].astype(str) + ' ' + data['Time'].astype(str))
    data['DateTime'] = datetimecol
    data.drop(['Date', 'Time'], axis=1, inplace=True)
    cols = data.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    data = data[cols]
    data['year'] = data['DateTime'].dt.year
    data['month'] = data['DateTime'].dt.month
    data['day'] = data['DateTime'].dt.day
    data['hour'] = data['DateTime'].dt.hour
    data.drop(['DateTime'], axis=1, inplace=True)
    data.drop(['year'], axis=1, inplace=True)

    # assume for now we're trying to predict the CO(GT) column

    X, Y = data.drop(columns=['CO(GT)']), data[['CO(GT)']]

    minmax_scaler = MinMaxScaler()
    X = minmax_scaler.fit_transform(X)
    Y = minmax_scaler.fit_transform(Y)
    return X,Y


if __name__ == "__main__":
    np.random.seed(123)
    X, Y = preprocess()
    n_feat = X.shape[1]

    optim_methods = ["NE", "GD"]
    learning_rate = 0.0001
    max_iter = 1000
    # optim_method = "GD"

    # Passive clients do not have outputs
    if rank != 1:
        Y = np.zeros(Y.shape)

    if rank == 0:
        features_owned = [-1]
    elif rank != n_cli and rank != 0:
        features_owned = np.arange(int((n_feat // n_cli) * (rank - 1)), int((n_feat // n_cli) * rank))
    else:
        features_owned = np.arange(int((n_feat // n_cli) * (rank - 1)), n_feat)
    for i in range(n_feat):
        if i not in features_owned:
            X[:, i] = np.zeros(X[:, i].shape)

    # print("rank:", rank, " X:", X[0], " Y:", Y)
    train_test_split_ratio = 0.8

    for winsz in [50, 100, 200, 400]:
        num_windows = len(Y) // winsz
        mse_sarimax = []
        for i in range(min(num_windows, 5)):
            X_train = X[int(i * winsz):int(i * winsz) + int(train_test_split_ratio * winsz)]
            X_test = X[int(i * winsz) + int(train_test_split_ratio * winsz):int(i * winsz) + winsz]
            Y_train = Y[int(i * winsz):int(i * winsz) + int(train_test_split_ratio * winsz)]
            Y_test = Y[int(i * winsz) + int(train_test_split_ratio * winsz):int(i * winsz) + winsz]
            order = None
            if rank == 1:
                auto_arima_res = auto_arima(Y_train, start_p=0, start_q=0, seasonal=False)
                (p, d, q), (P, D, Q, S) = auto_arima_res.order, auto_arima_res.seasonal_order
                order = [p, d, q, P, D, Q, S]
                for cli in range(0, n_cli + 1):
                    if cli != 1:
                        comm.send(order, dest=cli)

            elif rank != 1:
                [p, d, q, P, D, Q, S] = comm.recv(source=1)


            for optim in optim_methods:
                if optim == "NE":
                    continue
                    sarimax = SARIMAX_POLYPROCESSOR_VFL(p, d, q, P, D, Q, S, Y_train, X_train, n_cli, optim)
                elif optim == "GD":
                    sarimax = SARIMAX_POLYPROCESSOR_VFL(p, d, q, P, D, Q, S, Y_train, X_train, n_cli, optim, learning_rate, max_iter)
                sarimax.step1()
                sarimax.step2(X_train)
            # Y_predictions_sarimax_ne = sarimax.forecast(X_test, Y_test)
            #
            # final_preds = AGGSHR(Y_predictions_sarimax_ne, 1)
            # if rank == 1:
            #     print(final_preds+1)
            #     print(Y_test)

            # print("done")
            # exit()
