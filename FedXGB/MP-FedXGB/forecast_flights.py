import seaborn as sns
from numpy.polynomial import polynomial as poly
from statsmodels.tsa.tsatools import lagmat
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
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
    np.random.seed(123)
    data = sns.load_dataset('flights')
    monthLst = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    intLst = list(range(12))
    data = data.replace(monthLst, intLst)
    data['month'] = pd.to_numeric(data['month'])
    train_valid_ratio = 0.8
    X, Y = data.drop(columns=['passengers']), data[['passengers']]
    minmax_scaler = MinMaxScaler()
    X = minmax_scaler.fit_transform(X)
    Y = minmax_scaler.fit_transform(Y)
    train_test_split_ratio = 0.8

    for winsz in [60, 80, 100, 120, len(Y)]:
        num_windows = len(Y) // winsz
        mse_sarimax_ne = []
        mse_sarimax_mle = []
        mse_skforecast = []
        mse_lstm = []
        mse_saritrax = []
        for i in range(min(num_windows, 5)):
            X_train = X[int(i * winsz):int(i * winsz) + int(train_test_split_ratio * winsz)]
            X_test = X[int(i * winsz) + int(train_test_split_ratio * winsz):int(i * winsz) + winsz]
            Y_train = Y[int(i * winsz):int(i * winsz) + int(train_test_split_ratio * winsz)]
            Y_test = Y[int(i * winsz) + int(train_test_split_ratio * winsz):int(i * winsz) + winsz]

            # auto_arima_res = auto_arima(Y_train, seasonal=True, D=12)
            # (p, d, q), (P, D, Q, S) = auto_arima_res.order, auto_arima_res.seasonal_order
            (p, d, q), (P, D, Q, S) = (1, 1, 1), (1, 1, 1, 12)

            """SARIMAX NE"""
            sarimax = SARIMAX_POLYPROCESSOR(p, d, q, P, D, Q, S, Y_train, X_train)
            sarimax.step1()
            sarimax.step2(X_train)
            Y_predictions_sarimax_ne = sarimax.forecast(X_test, Y_test)
            mse_sarimax_ne.append(mean_squared_error(Y_test, Y_predictions_sarimax_ne))
            # plt.plot(Y_predictions_sarimax_ne, c='green', label='SARIMAX_NE')

            """SARIMAX MLE"""
            sarimodel = sm.tsa.statespace.SARIMAX(Y_train, exog=X_train, order=(p, d, q), seasonal_order=(P, D, Q, S))
            res = sarimodel.fit(disp=False)
            Y_predictions_sarimax_mle = res.forecast(len(Y_test), exog=X_test)
            # plt.plot(Y_predictions_sarimax_mle, c='blue', label='SARIMAX_MLE')
            mse_sarimax_mle.append(mean_squared_error(Y_test, Y_predictions_sarimax_mle))

            """LSTM"""
            poly = SARIMAX_POLYPROCESSOR(p, d, q, P, D, Q, S, Y_train, X_train)
            forecaster = ForecasterAutoreg(None, lags=max(len(poly.ar_poly) - 1, 1))
            x_window_train_lstm, y_window_train_lstm = forecaster.create_train_X_y(
                y=pd.Series(np.reshape(Y_train, (-1,))),
                exog=pd.DataFrame(X_train))
            x_window_valid = X_test
            x_window_train_lstm = x_window_train_lstm.to_numpy().reshape(
                (len(x_window_train_lstm), x_window_train_lstm.shape[1], 1))
            y_window_train_lstm = y_window_train_lstm.to_numpy().reshape((len(y_window_train_lstm), 1))
            lstm = Sequential()
            lstm.add(LSTM(64, return_sequences=True, input_shape=(None, 1)))
            lstm.add(LSTM(64, return_sequences=False))
            lstm.add(Dense(32))
            lstm.add(Dense(1))
            lstm.compile(optimizer='adam', loss='mean_squared_error')
            lstm.fit(x_window_train_lstm, y_window_train_lstm, epochs=500, batch_size=32, verbose=0)
            last_window_y = np.flip(y_window_train_lstm[-forecaster.lags.size:])
            curr_ind = 0
            predicts = []
            # while curr_ind < len(x_window_train_lstm):
            #     predicts.append(lstm.predict(np.reshape(x_window_train_lstm[curr_ind], (1, x_window_train_lstm.shape[1], 1)))[0])
            #     curr_ind += 1

            while curr_ind < len(x_window_valid):
                x_window_curr = x_window_valid[curr_ind].reshape((-1, 1))
                rec_window_valid_curr = np.concatenate((last_window_y, x_window_curr), axis=0)
                rec_window_valid_curr = np.reshape(rec_window_valid_curr,
                                                   (1, rec_window_valid_curr.shape[0], rec_window_valid_curr.shape[1]))
                predicts.append(lstm.predict(rec_window_valid_curr, verbose=0)[0])
                last_window_y = np.roll(last_window_y, 1)
                last_window_y[0] = predicts[-1]
                curr_ind += 1

            Y_predictions_LSTM = np.array(predicts)
            # plt.plot(Y_predictions_LSTM, c='red', label='LSTM')
            mse_lstm.append(mean_squared_error(Y_test, Y_predictions_LSTM))

            """Skforecast"""
            poly = SARIMAX_POLYPROCESSOR(p, d, q, P, D, Q, S, Y_train, X_train)
            xgreg = XGBRegressor(random_state=123, max_depth=5)
            xgreg.fit(X_train, Y_train)
            y_window_train_pred = xgreg.predict(X_train)

            forecaster = ForecasterAutoreg(
                regressor=XGBRegressor(random_state=123, max_depth=5),
                lags=max(1, len(poly.ar_poly) - 1)
            )

            forecaster.fit(y=pd.Series(np.reshape(Y_train, (-1,))), exog=pd.DataFrame(X_train))
            Y_predictions_skforecast = forecaster.predict(steps=len(Y_test),
                                                          exog=pd.DataFrame(X_test)).to_numpy()
            # plt.plot(Y_predictions_skforecast, 'y', label='Skforecast')

            mse_skforecast.append(mean_squared_error(Y_test, Y_predictions_skforecast))
            # plt.plot(Y_test, label="True output", c='black')
            # plt.legend()
            # plt.show()

        print("Prequential Window size:", winsz, "MSE SARIMAX NE:", mean(mse_sarimax_ne), "MSE SARIMAX MLE:",
              mean(mse_sarimax_mle), "MSE LSTM:", mean(mse_lstm), "MSE Skforecast:", mean(mse_skforecast), "MSE SARITRAX:", mean(mse_saritrax))