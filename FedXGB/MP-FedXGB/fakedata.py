import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller as adf
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from skforecast.ForecasterAutoreg import ForecasterAutoreg
import TimeSeriesFuncs as tsf
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    n = 365 * 11
    np.random.seed(0)
    linear, trend, seasonality, noise = [], [], [], []
    for i in range(n):
        linear.append(1)
        trend.append((i ** 1) * 0.001)
        seasonality.append(1.2 * math.sin(math.pi * i * 2 / 365))  ## 1 year as a cycle

    linear = np.array(linear)
    trend = np.array(trend)
    seasonality = np.array(seasonality)
    noise = (np.random.randn(n)).reshape(-1) * 0.3

    # In[24]:

    # f, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharey=True, figsize=(14, 10))
    # f.subplots_adjust(hspace=0.5)
    # ax1.plot(np.arange(n), linear)
    # ax2.plot(np.arange(n), trend)
    # ax3.plot(np.arange(n), seasonality)
    # ax4.plot(np.arange(n), noise)
    # ax1.set_title('Linear')
    # ax2.set_title('Trend')
    # ax3.set_title('Seasonality')
    # ax4.set_title('noise')

    # In[25]:

    y = linear + trend + seasonality + noise
    x = np.arange(n)

    # In[26]:

    # y=y.reshape(-1,1)
    x = x.reshape(-1, 1)

    # In[27]:

    # plt.figure(figsize=(10, 6))
    # plt.plot(x, y)
    # plt.title('Fake Data')

    # plt.show()
    # In[28]:

    n_valid = 365 * 4
    idx = np.arange(n)
    set_idx = set(idx)
    # valid_start = np.random.choice(idx[:-n_valid])
    valid_start = idx[-n_valid]
    valid_idx = idx[valid_start:valid_start + n_valid]
    set_val_idx = set(valid_idx)
    train_idx_set = set_idx.difference(set_val_idx)
    train_idx = np.array(list(train_idx_set))
    valid_idx = np.array(list(set_val_idx))
    x_train = x[train_idx]
    x_valid = x[valid_idx]
    y_train = y[train_idx]
    y_valid = y[valid_idx]


    # x_train = pd.DataFrame(x_train, columns=['Date'])
    # x_valid = pd.DataFrame(x_valid, columns=['Date'])

    # # forecaster = ForecasterAutoreg(
    # #     regressor=LinearRegression(),
    # #     lags=5
    # # )
    # #
    # # forecaster.fit(y = pd.Series(y_train), exog = pd.DataFrame(x_train, columns=['Date']))
    #

    # xgregressor = XGBRegressor(random_state = 123)
    # xgregressor.fit(x_train, detrended_train_x)
    #
    # y_pred_train = xgregressor.predict(x_train) + regressor.predict(x_train)
    #
    # remainders = x_valid % period
    #
    # closest_quotient = len(train_idx)//period
    #
    # output_indices = (period * (closest_quotient - 1) + remainders).reshape(-1)
    #
    #
    # y_pred_valid = xgregressor.predict(x_train[output_indices]) + regressor.predict(x_valid)
    #
    #
    #
    #
    # # # regressor = XGBRegressor(random_state=123, max_depth=5)
    # # # regressor.fit(x_train, y_train)
    # # # y_pred_train = regressor.predict(x_train).reshape(-1, 1)
    # plt.plot(train_idx, y_train, 'b.')
    # plt.plot(train_idx, y_pred_train, 'g.')
    # plt.plot(valid_idx, y_valid, 'r.')
    # plt.plot(valid_idx, y_pred_valid, 'g.')
    # print(mean_squared_error(y_valid, y_pred_valid))
    # plt.show()
    #
    # forecaster = ForecasterAutoreg(
    #     regressor=XGBRegressor(random_state=123, max_depth=5),
    #     lags=int(period)
    # )
    #
    # x_train = pd.DataFrame(x_train, columns=['Date'])
    # x_valid = pd.DataFrame(x_valid, columns=['Date'])
    #
    # xgreg = XGBRegressor(random_state=123, max_depth = 5)
    # xgreg.fit(x_train, y_train)
    # y_pred_train = xgreg.predict(x_train)
    #
    # forecaster.fit(y=pd.Series(y_train), exog=x_train)
    # y_pred_test = forecaster.predict(steps=len(y_valid), exog=x_valid)
    #
    # plt.plot(train_idx, y_train, 'b.')
    # plt.plot(train_idx, y_pred_train, 'g.')
    # # # # y_pred_valid = regressor.predict(x_valid).reshape(-1,1)
    # plt.plot(valid_idx, y_valid, 'r.')
    # plt.plot(valid_idx, y_pred_test, 'g.')
    # print(mean_squared_error(y_valid, y_pred_test))
    # plt.show()
    #differencing
    y_train_df = pd.DataFrame(y_train)
    diffs = 0
    while True:
        d1 = adf(y_train_df)
        if d1[1] <= 0.05:  # its stationary
            break
        else:
            y_train_df = y_train_df.diff()
            y_train_df.dropna(inplace=True)
            diffs += 1

    pcf = pacf(y_train_df, alpha=0.05)
    p = 0
    for i in range(1, len(pcf[0])):
        if pcf[0][i] >= pcf[1][i][0] and pcf[0][i] <= pcf[1][i][1]:
            p = i
            break

    q = 0
    af = acf(y_train, alpha=0.05)
    for i in range(1, len(pcf[0])):
        if af[0][i] >= af[1][i][0] and af[0][i] <= af[1][i][1]:
            q = i
            break

    period = np.argmax(np.abs(af[0][2:])) + 2

    # mod = sm.tsa.arima.ARIMA(endog, order=(p, d, q), seasonal_order=(P, D, Q, 365))
    # res = mod.fit(method='innovations_mle', low_memory=True, cov_type='none')
    sarimodel = sm.tsa.statespace.SARIMAX(y_train, exog=x_train, order=(p, diffs, q), seasonal_order=(p, diffs, q, 30))
    res = sarimodel.fit()
    y_pred_test = res.forecast(steps=len(x_valid), exog=x_valid)
    y_pred_train = res.predict(start=0, end=len(x_train) - 1, exog=x_train)
    plt.plot(train_idx, y_train, 'b')
    plt.plot(train_idx, y_pred_train, 'g')
    # # # y_pred_valid = regressor.predict(x_valid).reshape(-1,1)
    plt.plot(valid_idx, y_valid, 'r')
    plt.plot(valid_idx, y_pred_test, 'g')
    print(mean_squared_error(y_valid, y_pred_test))
    plt.show()

# In[ ]:
