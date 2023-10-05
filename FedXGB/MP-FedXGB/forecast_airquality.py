import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from pmdarima.arima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from SarimaPolynomial import SARIMAX_POLYPROCESSOR, SARITRAX
from keras.layers import Dense, LSTM
from keras import Sequential
import tensorflow as tf
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
    tf.random.set_seed(123)
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
    train_valid_ratio = 0.8

    X,Y = data.drop(columns=['CO(GT)']), data[['CO(GT)']]
    minmax_scaler = MinMaxScaler()
    X = minmax_scaler.fit_transform(X)
    Y = minmax_scaler.fit_transform(Y)
    train_test_split_ratio = 0.8


    for winsz in [50, 400]:
        num_windows = len(Y) // winsz
        num_windows = 1
        mse_sarimax_mle, mse_sarimax_ne, mse_lstm, mse_skforecast = [], [], [], []
        mape_sarimax_mle, mape_sarimax_ne, mape_lstm, mape_skforecast = [], [], [], []
        for i in range(min(num_windows, 5)):
            X_train = X[int(i * winsz):int(i * winsz) + int(train_test_split_ratio * winsz)]
            X_test = X[int(i * winsz) + int(train_test_split_ratio * winsz):int(i * winsz) + winsz]
            Y_train = Y[int(i * winsz):int(i * winsz) + int(train_test_split_ratio * winsz)]
            Y_test = Y[int(i * winsz) + int(train_test_split_ratio * winsz):int(i * winsz) + winsz]
            auto_arima_res = auto_arima(Y_train, start_p=0, start_q=0, seasonal=False)
            (p, d, q), (P, D, Q, S) = auto_arima_res.order, auto_arima_res.seasonal_order
            """SARIMAX NE"""
            sarimax = SARIMAX_POLYPROCESSOR(p, d, q, P, D, Q, S, Y_train, X_train)
            sarimax.step1()
            sarimax.step2(X_train)

            Y_predictions_sarimax_ne = sarimax.forecast(X_test, Y_test)
            mse_sarimax_ne.append(mean_squared_error(Y_test, Y_predictions_sarimax_ne))
            mape_sarimax_ne.append(mean_absolute_percentage_error(Y_test, Y_predictions_sarimax_ne))
            # print(mse_sarimax_ne[-1])

            plt.plot(Y_predictions_sarimax_ne, c='green', label='$STV_{L}(SARIMAX)$')

            """SARIMAX MLE"""
            sarimodel = sm.tsa.statespace.SARIMAX(Y_train, exog=X_train, order=(p, d, q), seasonal_order=(P, D, Q, S))
            res = sarimodel.fit(disp=False)
            Y_predictions_sarimax_mle = res.forecast(len(Y_test), exog=X_test)
            plt.plot(Y_predictions_sarimax_mle, c='blue', label='SARIMAX MLE')
            mse_sarimax_mle.append(mean_squared_error(Y_test, Y_predictions_sarimax_mle))
            mape_sarimax_mle.append(mean_absolute_percentage_error(Y_test, Y_predictions_sarimax_mle))

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
            # # while curr_ind < len(x_window_train_lstm):
            # #     predicts.append(lstm.predict(np.reshape(x_window_train_lstm[curr_ind], (1, x_window_train_lstm.shape[1], 1)))[0])
            # #     curr_ind += 1

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
            plt.plot(Y_predictions_LSTM, c='red', label='LSTM')
            mse_lstm.append(mean_squared_error(Y_test, Y_predictions_LSTM))
            # mape_lstm.append(mean_absolute_percentage_error(Y_test, Y_predictions_LSTM))

            """Skforecast"""
            poly = SARIMAX_POLYPROCESSOR(p, d, q, P, D, Q, S, Y_train, X_train)
            xgreg = XGBRegressor(random_state=123, max_depth=5)
            xgreg.fit(X_train, Y_train)
            y_window_train_pred = xgreg.predict(X_train)

            forecaster = ForecasterAutoreg(
                regressor=XGBRegressor(random_state=123, max_depth=3),
                lags=max(1, len(poly.ar_poly) - 1)
            )

            forecaster.fit(y=pd.Series(np.reshape(Y_train, (-1,))), exog=pd.DataFrame(X_train))
            Y_predictions_skforecast = forecaster.predict(steps=len(Y_test),
                                                          exog=pd.DataFrame(X_test)).to_numpy()
            plt.plot(Y_predictions_skforecast, c='orange', label='$STV_{T}$')
            #
            mse_skforecast.append(mean_squared_error(Y_test, Y_predictions_skforecast))
            mape_skforecast.append(mean_absolute_percentage_error(Y_test, Y_predictions_skforecast))
            plt.plot(Y_test, c='black', label="True output")
            # plt.legend()
            # plt.show()

            """Diffusion models"""
            filename = "diffusion_forecasts/forecast_airquality_diffusion_" + str(winsz) + ".npy"
            Y_predictions_diffusion = np.load(filename)
            plt.plot(Y_predictions_diffusion, label="$SSSD^{S4} Diffusion$", c="magenta")

            plt.rcParams.update({'font.size': 15})
            plt.ylabel("output value")
            plt.legend()
            imgfile = "forecast_plots/airquality_" + str(winsz) + ".pdf"
            plt.savefig(imgfile)
            plt.clf()

        print("Prequential Window size:", winsz, "MSE SARIMAX NE:", mean(mse_sarimax_ne), "MSE SARIMAX MLE:",
              mean(mse_sarimax_mle), "MSE LSTM:", mean(mse_lstm), "MSE Skforecast:", mean(mse_skforecast))
        # print("Prequential Window size:", winsz, "MAPE SARIMAX NE:", mean(mape_sarimax_ne), "MAPE SARIMAX MLE:",
        #       mean(mape_sarimax_mle), "MAPE LSTM:", mean(mape_lstm), "MAPE Skforecast:", mean(mape_skforecast))
        # plt.plot(y_window_train_lstm)
