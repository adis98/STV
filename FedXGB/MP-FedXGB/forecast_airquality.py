import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import TimeSeriesFuncs as tsf
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import adfuller as adf
from statsmodels.tsa.arima.model import ARIMA
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose
from sdt.changepoint import Pelt
from tensorflow import keras
from tensorflow.keras.layers import Dense,LSTM,Dropout,Flatten
from tensorflow.keras import Sequential


def mean(L):
    if len(L) == 0:
        return None
    else:
        return float(sum(L)) / len(L)


if __name__ == "__main__":
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

    # assume for now we're trying to predict the CO(GT) column
    train_valid_ratio = 0.7

    x, y = data.iloc[:, 1:], data.iloc[:, 0]

    train_samples = int(train_valid_ratio * len(data))
    train_idx = np.arange(train_samples)
    valid_idx = np.arange(train_samples, len(data))
    x_train, x_valid = x.iloc[:train_samples].to_numpy(), x.iloc[train_samples:].to_numpy()
    y_train, y_valid = y.iloc[:train_samples].to_numpy(), y.iloc[train_samples:].to_numpy()
    x, y = x.to_numpy(), y.to_numpy()
    y_train_df = pd.DataFrame(y_train)
    d = 0
    while True:
        diff = adf(y_train_df)
        if diff[1] <= 0.05:  # its stationary
            break
        else:
            y_train_df = y_train_df.diff()
            y_train_df.dropna(inplace=True)
            d += 1

    pcf = pacf(y_train_df, alpha=0.05)
    p = 0
    for i in range(1, len(pcf[0])):
        if pcf[0][i] >= pcf[1][i][0] and pcf[0][i] <= pcf[1][i][1]:
            p = i
            break

    q = 0
    af = acf(y_train_df, alpha=0.05)
    for i in range(1, len(pcf[0])):
        if af[0][i] >= af[1][i][0] and af[0][i] <= af[1][i][1]:
            q = i
            break

    peaks = find_peaks(af[0])
    values = af[0][peaks[0]]
    primary_period = peaks[0][np.argmax(values)]

    # decompose = seasonal_decompose(y_train, period=primary_period)
    det = Pelt(cost="l2", min_size=len(y_train), jump=1)
    chgpts = det.find_changepoints(y_train, 1)

    window_start = 0
    current_window_end_index = 0
    # window_end = chgpts[current_window_end_index]
    mse_valid_dec = []
    mse_valid_sarimax = []
    mse_valid_skforecast = []
    mse_valid_lstm = []
    while current_window_end_index <= min(5, len(chgpts)):
        if current_window_end_index == len(chgpts):  # last window
            y_window = y[window_start:]
            x_window = x[window_start:]
        else:
            y_window = y[window_start:chgpts[current_window_end_index]]
            x_window = x[window_start:chgpts[current_window_end_index]]

        x_window_train = x_window[:int(train_valid_ratio * len(x_window))]
        x_window_valid = x_window[int(train_valid_ratio * len(x_window)):]
        y_window_train = y_window[:int(train_valid_ratio * len(y_window))]
        y_window_valid = y_window[int(train_valid_ratio * len(y_window)):]

        """DECOMPOSED METHOD WITH SARIMA TREND MODEL AND STANDARD XGB"""
        decompose_window_train = seasonal_decompose(y_window_train, period=primary_period)
        y_window_train_sarimax_dec = decompose_window_train.trend + decompose_window_train.seasonal - decompose_window_train.seasonal
        y_window_train_tree_dec = decompose_window_train.resid[~np.isnan(decompose_window_train.resid)] + decompose_window_train.seasonal[~np.isnan(decompose_window_train.resid)]
        x_window_train_tree = x_window_train[~np.isnan(decompose_window_train.resid)]
        fv = pd.DataFrame(y_window_train_sarimax_dec).first_valid_index()
        lv = pd.DataFrame(y_window_train_sarimax_dec).last_valid_index()
        y_window_train_sarimax = y_window_train_sarimax_dec[~np.isnan(y_window_train_sarimax_dec)]
        sarimodel = sm.tsa.statespace.SARIMAX(y_window_train_sarimax, order=(p, d, q),
                                              seasonal_order=(p, d, q, primary_period))
        xgreg = XGBRegressor(random_state=123, max_depth=5)
        xgreg.fit(x_window_train_tree, y_window_train_tree_dec)
        res = sarimodel.fit()
        sari_preds_1 = res.predict(start=-fv, end=-1)
        sari_preds_2 = res.predict(start=0, end=lv - fv)
        sari_preds_3 = res.predict(start=lv - fv + 1, end=len(decompose_window_train.trend) - 1 - fv)
        y_window_train_pred_sari = np.concatenate((sari_preds_1, sari_preds_2, sari_preds_3))
        y_window_train_pred_tree = xgreg.predict(x_window_train)
        y_window_train_pred = y_window_train_pred_sari + y_window_train_pred_tree
        y_window_valid_pred_sari = res.predict(start=len(decompose_window_train.trend)-fv,
                                               end=len(decompose_window_train.trend) + len(y_window_valid) - 1 - fv)
        y_window_valid_pred_tree = xgreg.predict(x_window_valid)
        y_window_valid_pred = y_window_valid_pred_sari + y_window_valid_pred_tree + np.mean(y_window_train - y_window_train_pred) #- y_window_valid_pred_tree
        mse_valid_dec.append(mean_squared_error(y_window_valid, y_window_valid_pred))
        plt.plot(y_window_valid, 'b')
        plt.plot(y_window_valid_pred, 'g')

        """SARIMAX WITH EXOGENOUS"""
        # sarimodel = sm.tsa.statespace.SARIMAX(y_window_train, exog=x_window_train, order=(p, d, q),
        #                                       seasonal_order=(p, d, q, primary_period))
        # res = sarimodel.fit()
        # y_window_train_pred = res.predict(start=0, end=len(x_window_train) - 1, exog=x_window_train)
        # y_window_valid_pred = res.predict(start=len(x_window_train), end=len(x_window_train) + len(x_window_valid) - 1,
        #                                   exog=x_window_valid)
        # mse_valid_sarimax.append(mean_squared_error(y_window_valid, y_window_valid_pred))
        # plt.plot(y_window_valid_pred, 'r')
    #
    #     """SKFORECAST"""
    #     xgreg = XGBRegressor(random_state=123, max_depth=5)
    #     xgreg.fit(x_window_train, y_window_train)
    #     y_window_train_pred = xgreg.predict(x_window_train)
    #
    #     forecaster = ForecasterAutoreg(
    #         regressor=XGBRegressor(random_state=123, max_depth=3),
    #         lags=int(primary_period)
    #     )
    #
    #     forecaster.fit(y=pd.Series(y_window_train), exog=pd.DataFrame(x_window_train))
    #     y_window_valid_pred = forecaster.predict(steps=len(y_window_valid),
    #                                              exog=pd.DataFrame(x_window_valid)).to_numpy()
    #     mse_valid_skforecast.append(mean_squared_error(y_window_valid, y_window_valid_pred))
    # #     # plt.plot(y_window_valid_pred, 'y')
    # #
    #     """LSTM"""
    #     forecaster = ForecasterAutoreg(None, lags=int(primary_period))
    #     x_window_train_lstm, y_window_train_lstm = forecaster.create_train_X_y(y=pd.Series(y_window_train), exog=pd.DataFrame(x_window_train))
    #     x_window_train_lstm = x_window_train_lstm.to_numpy().reshape((len(x_window_train_lstm), x_window_train_lstm.shape[1], 1))
    #     y_window_train_lstm = y_window_train_lstm.to_numpy().reshape((len(y_window_train_lstm), 1))
    #     lstm = Sequential()
    #     lstm.add(LSTM(128, return_sequences=True, input_shape=(None, 1)))
    #     lstm.add(LSTM(64, return_sequences=False))
    #     lstm.add(Dense(25))
    #     lstm.add(Dense(1))
    #     lstm.compile(optimizer='adam', loss='mean_squared_error')
    #     lstm.fit(x_window_train_lstm, y_window_train_lstm, epochs=50, batch_size=32)
    #     last_window_y = np.flip(y_window_train_lstm[-primary_period:])
    #     curr_ind = 0
    #     predicts = []
    #     while curr_ind < len(x_window_valid):
    #         x_window_curr = x_window_valid[curr_ind].reshape((-1, 1))
    #         rec_window_valid_curr = np.concatenate((last_window_y, x_window_curr), axis=0)
    #         rec_window_valid_curr = np.reshape(rec_window_valid_curr, (1, rec_window_valid_curr.shape[0], rec_window_valid_curr.shape[1]))
    #         predicts.append(lstm.predict(rec_window_valid_curr)[0])
    #         last_window_y = np.roll(last_window_y, 1)
    #         last_window_y[0] = predicts[-1]
    #         curr_ind += 1
    #
    #     y_window_valid_pred = np.array(predicts)
    #     mse_valid_lstm.append(mean_squared_error(y_window_valid, y_window_valid_pred))
    #     # plt.plot(y_window_valid_pred, 'k')
        if current_window_end_index < len(chgpts):
            window_start = chgpts[current_window_end_index]
        current_window_end_index += 1

        plt.show()
    #
    print(mean(mse_valid_dec), mean(mse_valid_sarimax), mean(mse_valid_skforecast), mean(mse_valid_lstm))
    # sarimodel = sm.tsa.statespace.SARIMAX(y_train, order=(p, d, q), seasonal_order=(p, d, q, primary_period))

    # arimodel = ARIMA(trend_data_np, order=(p, d, q))
    # res = arimodel.fit()
    # res = sarimodel.fit()
    # arima_train_pred = res.predict(start=0, end=len(x_train) - 1)
    # sarima_train_pred = res.predict(start=0, end=len(x_train) - 1)
    # xgreg_dec = XGBRegressor(random_state=123, max_depth=5)
    # # xgreg_dec.fit(x_train, y_train - arima_train_pred)
    # xgreg_dec.fit(x_train, y_train - sarima_train_pred)
    # # y_pred_train = xgreg_dec.predict(x_train) + arima_train_pred
    # y_pred_train = xgreg_dec.predict(x_train) + sarima_train_pred
    #
    # max_train, min_train = np.max(x_train, axis=0), np.min(x_train, axis=0)
    #
    # remainders = valid_idx % period
    #
    # closest_quotient = len(train_idx) // period
    #
    # output_indices = (period * (closest_quotient - 1) + remainders).reshape(-1)
    #
    # y_pred_test = xgreg_dec.predict(x_valid) + res.predict(start=len(x_train), end=len(x_train) + len(x_valid) - 1)
    # print(r2_score(y_valid, y_pred_test))
    # plt.plot(train_idx, y_train, 'b')
    # plt.plot(train_idx, y_pred_train, 'g')
    # plt.plot(valid_idx, y_valid, 'r')
    # plt.plot(valid_idx, y_pred_test, 'g')
    # plt.show()
