import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from pmdarima.arima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from SarimaPolynomial import SARIMAX_POLYPROCESSOR
from keras.layers import Dense, LSTM
from keras import Sequential
import statsmodels.api as sm
from xgboost import XGBRegressor
import tensorflow as tf
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
    data_power = pd.read_csv('PVPower/Plant_1_Generation_Data.csv')
    data_power['DATE_TIME'] = pd.to_datetime(data_power['DATE_TIME'])
    data_weather = pd.read_csv('PVPower/Plant_1_Weather_Sensor_Data.csv')
    data_weather['DATE_TIME'] = pd.to_datetime(data_weather['DATE_TIME'])
    data_weather = data_weather.sort_values('DATE_TIME')
    data_power = data_power.sort_values('DATE_TIME')
    merged_data = pd.merge_asof(data_weather, data_power, on="DATE_TIME")
    merged_data.dropna(inplace=True)
    merged_data.drop(
        columns=['DC_POWER', 'DATE_TIME', 'PLANT_ID_x', 'SOURCE_KEY_x', 'PLANT_ID_y', 'SOURCE_KEY_y', 'TOTAL_YIELD'],
        inplace=True)
    Y, X = merged_data[['AC_POWER']], merged_data.drop(columns=['AC_POWER'])
    minmax_scaler = MinMaxScaler()
    X = minmax_scaler.fit_transform(X)
    Y = minmax_scaler.fit_transform(Y)
    train_test_split_ratio = 0.8

    for winsz in [25, 200]:
        num_windows = len(Y) // winsz
        num_windows = 1
        mse_sarimax_ne = []
        mse_sarimax_mle = []
        mse_skforecast = []
        mse_lstm = []
        # mse_sarimax_sgd = []
        for i in range(min(num_windows, 5)):
            np.random.seed(123)
            tf.random.set_seed(123)
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
            plt.plot(Y_predictions_sarimax_ne, c='green', label='$STV_{L}(SARIMAX)$')

            """SARIMAX MLE"""
            sarimodel = sm.tsa.statespace.SARIMAX(Y_train, exog=X_train, order=(p, d, q), seasonal_order=(P, D, Q, S))
            res = sarimodel.fit(disp=False)
            Y_predictions_sarimax_mle = res.forecast(len(Y_test), exog=X_test)
            plt.plot(Y_predictions_sarimax_mle, c='blue', label='SARIMAX MLE')
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
            plt.plot(Y_predictions_LSTM, c='red', label='LSTM')
            mse_lstm.append(mean_squared_error(Y_test, Y_predictions_LSTM))

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

            """Diffusion models"""
            filename = "diffusion_forecasts/forecast_pvpower_diffusion_" + str(winsz) + ".npy"
            Y_predictions_diffusion = np.load(filename)
            plt.plot(Y_predictions_diffusion, label="$SSSD^{S4} Diffusion$", c="magenta")
            plt.plot(Y_test, label="True output", c="black")

            plt.rcParams.update({'font.size': 15})
            plt.ylabel("output value")
            plt.legend()
            imgfile = "forecast_plots/pvpower_" + str(winsz) + ".pdf"
            plt.savefig(imgfile)
            plt.clf()

            mse_skforecast.append(mean_squared_error(Y_test, Y_predictions_skforecast))

        print("Prequential Window size:", winsz, "MSE SARIMAX NE:", mean(mse_sarimax_ne), "MSE SARIMAX MLE:",
              mean(mse_sarimax_mle), "MSE LSTM:", mean(mse_lstm), "MSE Skforecast:", mean(mse_skforecast))