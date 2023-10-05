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
import dateutil.parser
import tensorflow as tf

warnings.filterwarnings("ignore")


def mean(L):
    if len(L) == 0:
        return None
    else:
        return float(sum(L)) / len(L)


if __name__ == "__main__":
    np.random.seed(42)
    tf.random.set_seed(42)
    data = pd.read_csv('ASMLData/filtered_data_complete_chuck1.csv')

    data['Time'] = pd.to_datetime(data['Time'])

    data['k6'] = ((data['rs_c00'] - data['ra_c00'])/2) * (10**8)
    data['k3'] = ((data['ms_c00'] + data['ma_c00'])/2) * (10**7)
    data['k7'] = data['k7'] * (10**5)
    data['k12'] = data['k12'] * (10**5)
    data.drop(columns=['rs_c00', 'ra_c00', 'ms_c00', 'ma_c00'], inplace=True)
    # plt.plot(data['k12'])
    # plt.show()
    # exit()
    target_col = 'k12'
    # seg_1_time = '2019-12-10'
    seg_1_time = '2020-06-24'
    seg_1 = data[data.Time < dateutil.parser.parse(seg_1_time)]
    # seg_1.plot(x='Time', y='k12')
    # seg_1.plot(x='Time', y='KdPSMD2yChuck1')

    seg_2_time = '2021-06-03'
    # seg_2_time = '2021-05-29'
    seg_2 = data[(data.Time > dateutil.parser.parse(seg_1_time)) &
                 (data.Time < dateutil.parser.parse(seg_2_time))]

    # seg_2.plot(x='Time', y='k12')
    # seg_2.plot(x='Time', y='KdPSMD2yChuck1')

    seg_3_time = '2021-12-11'
    seg_3 = data[(data.Time > dateutil.parser.parse(seg_2_time)) &
                 (data.Time < dateutil.parser.parse(seg_3_time))]

    # seg_3.plot(x='Time', y='k12')
    # seg_3.plot(x='Time', y='KdPSMD2yChuck1')
    bmmo_paramameters = ['k3', 'k6', 'k7', 'k12', 'k13']
    paris_parameters= ['KdPSAMeanMagChuck1','KdPSAMeanRotChuck1','KdPSMD2xChuck1',
                       'KdPSMD2yChuck1','KdPSMD3xChuck1']
    #
    # for i in range(len(bmmo_paramameters)):
    #     plt.figure(i)
    #     plt.subplot(211)
    #     plt.plot(seg_2[bmmo_paramameters[i]], label=bmmo_paramameters[i])
    #     plt.legend()
    #     plt.subplot(212)
    #     plt.plot(seg_2[paris_parameters[i]], label=paris_parameters[i])
    #     plt.legend()
    #
    # plt.show()
    # exit()

    addn_cols = []
    for col in bmmo_paramameters:
        if col != target_col:
            addn_cols.append(col)
    seg_2.drop(columns=['Unnamed: 0', 'Time', 'sample_dt']+addn_cols, inplace=True)
    corrcoefs_seg_2 = pd.DataFrame(columns=["parameter", "corrcoef"])
    for column in seg_2.columns:
        if column not in ['Time', 'sample_dt', 'Unnamed: 0']:
            corrcoef = np.corrcoef(np.reshape(seg_2[column], (-1,)), np.reshape(seg_2[target_col], (-1,)))[0, 1]
            corrcoefs_seg_2 = corrcoefs_seg_2.append({"parameter": column, "corrcoef": corrcoef}, ignore_index=True)
    # corrcoefs_seg_2.plot.barh(x='parameter', y='corrcoef')
    # plt.show()
    # plt.clf()
    threshold = 0.7
    seg_2_exog_columns = []
    for i in range(len(corrcoefs_seg_2)):
        if abs(corrcoefs_seg_2.iloc[i, 1]) >= threshold and corrcoefs_seg_2.iloc[i, 0] != target_col:
            seg_2_exog_columns.append(corrcoefs_seg_2.iloc[i, 0])
    X, Y = seg_2[seg_2_exog_columns], seg_2[[target_col]]
    minmax_scaler = MinMaxScaler()
    X = minmax_scaler.fit_transform(X)
    Y = minmax_scaler.fit_transform(Y)
    train_test_split_ratio = 0.8
    for winsz in [25, 100]:
        num_windows = len(Y) // winsz
        num_windows = 1
        mse_sarimax_ne = []
        mse_sarimax_mle = []
        mse_skforecast = []
        mse_lstm = []
        mse_saritrax = []
        mape_sarimax_ne = []
        mape_sarimax_mle = []
        mape_skforecast = []
        mape_lstm = []
        for i in range(min(num_windows, 5)):
            X_train = X[int(i * winsz):int(i * winsz) + int(train_test_split_ratio * winsz)]
            X_test = X[int(i * winsz) + int(train_test_split_ratio * winsz):int(i * winsz) + winsz]
            Y_train = Y[int(i * winsz):int(i * winsz) + int(train_test_split_ratio * winsz)]
            Y_test = Y[int(i * winsz) + int(train_test_split_ratio * winsz):int(i * winsz) + winsz]

            auto_arima_res = auto_arima(Y_train)
            (p, d, q), (P, D, Q, S) = auto_arima_res.order, auto_arima_res.seasonal_order
            if target_col=='k12':
                (d, q) = (1,1)
            """SARIMAX NE"""
            sarimax = SARIMAX_POLYPROCESSOR(p, d, q, P, D, Q, S, Y_train, X_train)
            sarimax.step1()
            sarimax.step2(X_train)
            Y_predictions_sarimax_ne = sarimax.forecast(X_test, Y_test)
            mse_sarimax_ne.append(mean_squared_error(Y_test, Y_predictions_sarimax_ne))
            mape_sarimax_ne.append(mean_absolute_percentage_error(Y_test, Y_predictions_sarimax_ne))
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
            mape_lstm.append(mean_absolute_percentage_error(Y_test, Y_predictions_LSTM))

            """Skforecast"""
            poly = SARIMAX_POLYPROCESSOR(p, d, q, P, D, Q, S, Y_train, X_train)
            forecaster = ForecasterAutoreg(
                regressor=XGBRegressor(random_state=123, max_depth=5),
                lags=max(1, len(poly.ar_poly) - 1)
            )

            forecaster.fit(y=pd.Series(np.reshape(Y_train, (-1,))), exog=pd.DataFrame(X_train))
            exog_df = pd.DataFrame(X_test)
            exog_df.index = exog_df.index + len(X_train)
            Y_predictions_skforecast = forecaster.predict(steps=len(Y_test),
                                                          exog=exog_df).to_numpy()
            plt.plot(Y_predictions_skforecast, c='orange', label='$STV_{T}$')

            mse_skforecast.append(mean_squared_error(Y_test, Y_predictions_skforecast))

            """Diffusion models"""
            filename = "diffusion_forecasts/forecast_asml_diffusion_" + str(winsz) + ".npy"
            Y_predictions_diffusion = np.load(filename)
            plt.plot(Y_predictions_diffusion, label="$SSSD^{S4} Diffusion$", c="magenta")
            plt.plot(Y_test, label="True output", c="black")

            plt.rcParams.update({'font.size': 18})
            plt.rcParams.update({'axes.labelsize': 18})
            plt.ylabel("output value")
            plt.legend()
            imgfile = "forecast_plots/asml_" + str(winsz) + ".pdf"
            plt.savefig(imgfile)
            plt.clf()

        print("Prequential Window size:", winsz, "MSE SARIMAX NE:", mean(mse_sarimax_ne), "MSE SARIMAX MLE:",
              mean(mse_sarimax_mle), "MSE LSTM:", mean(mse_lstm), "MSE Skforecast:", mean(mse_skforecast))

        # print("Prequential Window size:", winsz, "MAPE SARIMAX NE:", mean(mape_sarimax_ne), "MAPE SARIMAX MLE:",
        #       mean(mape_sarimax_mle), "MAPE LSTM:", mean(mape_lstm), "MAPE Skforecast:", mean(mape_skforecast))