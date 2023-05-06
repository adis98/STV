import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from pmdarima.arima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from SarimaPolynomial import SARIMAX_POLYPROCESSOR, SARITRAX
from keras.layers import Dense, LSTM
from keras import Sequential
import statsmodels.api as sm
from xgboost import XGBRegressor
import warnings
from skforecast.ForecasterAutoreg import ForecasterAutoreg

warnings.filterwarnings("ignore")


def mean(L):
    if len(L) == 0:
        return None
    else:
        return float(sum(L)) / len(L)


if __name__ == "__main__":
    data_train = pd.read_csv("GarmentWorkerProductivity/garments_worker_productivity.csv")
    print()
