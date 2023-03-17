import seaborn as sns
import matplotlib.pyplot as plt
import TimeSeriesFuncs as tsf
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller as adf
from statsmodels.tsa.stattools import acf, pacf
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from scipy.signal import find_peaks
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.linear_model import LinearRegression
from sdt.changepoint import Pelt
from statsmodels.graphics.tsaplots import plot_pacf



if __name__ == "__main__":
    data = sns.load_dataset('flights')
    data.drop(columns=['year', 'month'], inplace=True)
    train_valid_ratio = 0.7
    train_samples = int(train_valid_ratio * len(data))
    x_train, x_valid = np.arange(0, train_samples), np.arange(train_samples, len(data))
    y_train, y_valid = data.iloc[:train_samples, ].to_numpy().reshape((-1)), data.iloc[train_samples:, ].to_numpy().reshape((-1))
    train_idx = np.arange(train_samples)
    valid_idx = np.arange(train_samples, len(data))

    # differencing
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

    af = acf(y_train_df, alpha=0.05)
    peaks = find_peaks(af[0])
    values = af[0][peaks[0]]
    primary_period = peaks[0][np.argmax(values)]

    decompose = seasonal_decompose(y_train, period=primary_period)

    seasonal_component = decompose.seasonal[~np.isnan(decompose.seasonal)]
    D = d
    # while True:
    #     diff = adf(seasonal_component)
    #     if diff[1] <= 0.05:  # its stationary
    #         break
    #     else:
    #         seasonal_component = seasonal_component[primary_period:] - seasonal_component[:-primary_period]
    #         D += 1

    #To get P and Q seasonally difference the data
    # plot_pacf(y_train_df)
    # pt
    pcf = pacf(y_train_df, alpha=0.05)
    P = 0
    for i in range(1, len(pcf[0])):
        if pcf[0][i] >= pcf[1][i][0] and pcf[0][i] <= pcf[1][i][1]:
            P = i
            break

    Q = 0
    for i in range(1, len(pcf[0])):
        if af[0][i] >= af[1][i][0] and af[0][i] <= af[1][i][1]:
            Q = i
            break

    #now to get p and q
    pcf = pacf(y_train_df[:primary_period], alpha=0.05)
    p = 0
    for i in range(1, len(pcf[0])):
        if pcf[0][i] >= pcf[1][i][0] and pcf[0][i] <= pcf[1][i][1]:
            p = i
            break

    q = 0
    af = acf(y_train_df[:primary_period], alpha=0.05)
    for i in range(1, len(pcf[0])):
        if af[0][i] >= af[1][i][0] and af[0][i] <= af[1][i][1]:
            q = i
            break




    """SARIMAX MODEL"""
    sarimodel = sm.tsa.statespace.SARIMAX(y_train, exog=x_train, order=(p, d, q), seasonal_order=(P, D, Q, primary_period))
    res = sarimodel.fit()
    y_pred_test = res.predict(start=len(x_train), end=len(x_train)+len(x_valid) - 1, exog=x_valid)
    # y_pred_train = res.predict(start=0, end=len(x_train) - 1, exog=x_train)
    # plt.plot(train_idx, y_train, 'b')
    # plt.plot(train_idx, y_pred_train, 'g')
    plt.plot(valid_idx, y_valid, 'r')
    plt.plot(valid_idx, y_pred_test, 'g')
    plt.show()

    """SKFORECAST MODEL"""
    # xgreg = XGBRegressor(random_state=123, max_depth=5)
    # xgreg.fit(np.reshape(x_train, (-1,1)), np.reshape(y_train, (-1,1)))
    # y_pred_train = xgreg.predict(x_train)
    #
    # forecaster = ForecasterAutoreg(
    #         regressor=XGBRegressor(random_state=123, max_depth=3),
    #         lags=int(primary_period)
    # )
    #
    # # forecaster.fit(y=pd.Series(y_train), exog=pd.DataFrame(x_train))
    # forecaster.fit(y=pd.Series(y_train))
    # # y_pred_test = forecaster.predict(steps=len(y_valid), exog=pd.DataFrame(x_valid))
    # y_pred_test = forecaster.predict(steps=len(y_valid))
    #
    # plt.plot(train_idx, y_train, 'b')
    # plt.plot(train_idx, y_pred_train, 'g')
    # plt.plot(valid_idx, y_valid, 'r')
    # plt.plot(valid_idx, y_pred_test, 'g')
    # plt.show()


    """DECOMPOSED MODEL"""
    trend_data = pd.DataFrame(y_train) - y_train_df
    trend_data = pd.DataFrame(decompose.trend)
    trend_data.dropna(inplace=True)
    det = Pelt(cost="l2", min_size=25, jump=1)
    chgpts = det.find_changepoints(trend_data.to_numpy(), 1)
    plt.plot(trend_data)
    plt.plot(chgpts, np.ones(len(chgpts)))
    plt.show()
    pass
    # forecaster = ForecasterAutoreg(LinearRegression(), lags=d)
    # forecaster.fit(pd.Series(trend_data.to_numpy().reshape(-1)))
    # y_pred_train_trend = forecaster.predict(steps = len(trend_data) - d, last_window=pd.Series(trend_data[:d].to_numpy().reshape(-1)))
    # trend_data_np = trend_data.to_numpy().reshape(-1)
    # y_pred_train_trend_np = y_pred_train_trend.to_numpy()
    #
    # y_pred_train_trend_np = np.concatenate((trend_data_np[:d], y_pred_train_trend_np))
    # y_pred_train_trend_end = forecaster.predict(steps = int(primary_period/2)).to_numpy().reshape(-1)
    # y_pred_train_trend_np = np.concatenate((y_pred_train_trend_np, y_pred_train_trend_end))
    # y_pred_train_trend_np = np.concatenate((y_train[:int(primary_period/2)], y_pred_train_trend_np))
    #
    #
    # # trend_data_np = np.append(y_train[:int(period/2)], trend_data.to_numpy().reshape((-1)))
    # # plt.plot(trend_data_np)
    # # plt.show()
    # # exit()
    # # sarimodel = sm.tsa.statespace.SARIMAX(y_train, order=(p, d, q), seasonal_order=(p, d, q, primary_period))
    # # sarimodel = sm.tsa.statespace.SARIMAX(trend_data)
    # # arimodel = ARIMA(trend_data_np, order=(p, d, q))
    # # arimodel = ARIMA(trend_data_np)
    # # res = arimodel.fit()
    # # res = sarimodel.fit()
    # # arima_train_pred = res.predict(start=0, end=len(x_train) - 1)
    # # sarima_train_pred = res.predict(start=0, end=len(x_train) - 1)
    # # plt.plot(sarima_train_pred)
    # # plt.show()
    # # exit()
    # # xgreg_dec = XGBRegressor(random_state=123, max_depth=5)
    # # xgreg_dec.fit(np.reshape(x_train, (-1,1)), y_train - arima_train_pred)
    # # xgreg_dec.fit(np.reshape(x_train, (-1, 1)), y_train - sarima_train_pred)
    # # xgreg_dec.fit(np.reshape(x_train, (-1, 1)), decompose.seasonal)
    # # y_pred_train = xgreg_dec.predict(x_train) + arima_train_pred
    # # y_pred_train = xgreg_dec.predict(x_train) + y_pred_train_trend_np
    # xgforecast = ForecasterAutoreg(XGBRegressor(random_state=123, max_depth=5), lags=int(primary_period))
    # xgforecast.fit(pd.Series(decompose.seasonal), exog=pd.DataFrame(np.reshape(x_train, (-1,1))))
    #
    #
    #
    # remainders = x_valid % primary_period
    #
    # closest_quotient = len(train_idx)//primary_period
    #
    # output_indices = (period * (closest_quotient - 1) + remainders).reshape(-1)
    #
    # # y_pred_test = xgreg_dec.predict(x_train[output_indices]) + res.predict(start=len(x_train), end=len(x_train)+len(x_valid)-1)
    # # y_pred_test = xgreg_dec.predict(x_train[output_indices]) + forecaster.predict(steps=len(y_valid)).to_numpy()
    # y_pred_test = xgforecast.predict(steps=len(y_valid), exog=pd.DataFrame(np.reshape(x_valid, (-1,1)))) + forecaster.predict(steps=len(y_valid)).to_numpy()
    #
    # y_pred_train = np.concatenate((decompose.seasonal[:int(primary_period)], xgforecast.predict(steps=len(y_train) - int(primary_period),
    #                                   exog=pd.DataFrame(np.reshape(x_train[int(primary_period):], (-1, 1))), last_window=pd.Series(y_train[:int(primary_period)])))) + y_pred_train_trend_np
    #
    # plt.plot(train_idx, y_train, 'b')
    # plt.plot(train_idx, y_pred_train, 'g')
    # plt.plot(valid_idx, y_valid, 'r')
    # plt.plot(valid_idx, y_pred_test, 'g')
    # plt.show()


