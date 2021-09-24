import numpy as np
import pandas as pd


def synthetic_data(Nt=2000, tf=80 * np.pi):
    '''
    create synthetic time series dataset
    : param Nt:       number of time steps
    : param tf:       final time
    : return t, y:    time, feature arrays
    '''
    t = np.linspace(0., tf, Nt)
    y = np.sin(2. * t) + 0.5 * np.cos(t) + np.random.normal(0., 0.2, Nt)

    return t, y


def main():
    prediction_horizon = 30
    val_size = 300
    total_train = 1500
    total = total_train + prediction_horizon + val_size

    simple = True

    if simple:
        time_idx = np.arange(total) / total * 2 * np.pi
        data_train = np.sin(2 * np.pi * 4 * time_idx)[:val_size + total_train]
        data_val = np.sin(2 * np.pi * 4 * time_idx)[total_train:total_train +
                                                    val_size]
        data_test = np.sin(2 * np.pi * 4 * time_idx)[total_train + val_size:]

    else:
        t, data_total = synthetic_data(Nt=total)
        data_train = data_total[:val_size + total_train]
        data_val = data_total[total_train:total_train + val_size]
        data_test = data_total[total_train + val_size:]

    df_train = {'data': data_train}
    df_train = pd.DataFrame(df_train)

    df_val = {'data': data_val}
    df_val = pd.DataFrame(df_val)

    df_test = {'data': data_test}
    df_test = pd.DataFrame(df_test)

    df_train.to_csv("train_series.csv")
    df_val.to_csv("val_series.csv")
    df_test.to_csv("test_series.csv")


if __name__ == '__main__':
    main()