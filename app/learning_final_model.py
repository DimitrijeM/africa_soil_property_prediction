import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import MinMaxScaler

from app import remove_m_co2_spectar, remove_rows_with_outlier, transform_m_columns_pca

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.random.seed(1)
np.set_printoptions(formatter={'float_kind': lambda x: "%.3f" % x})


def mean_columnwise_root_mse(y, y_pred):
    return np.sum((y - y_pred)**2/y.shape[0])/y.shape[1]


if __name__ == '__main__':
    df = pd.read_csv('../data/training.csv').set_index('PIDN')
    df.Depth = df.Depth.replace({'Subsoil': -1, 'Topsoil': 1})
    print(f"Ddataset shape: {df.shape}")
    target_columns = ['Ca', "P", "pH", "SOC", "Sand"]
    measured_columns = ['Depth']
    spatial_columns = [x for x in list(df.columns) if x not in (target_columns+measured_columns) and not x.startswith('m')]
    absorbance_measured_columns = [x for x in list(df.columns) if x not in target_columns and x.startswith('m')]
    print(f"Ciljni atributi: {len(target_columns)}. Prostorni atributi: {len(spatial_columns)}. "
          f"Izmereni atributi: {len(measured_columns)}. "
          f"Atributi o izmerenoj apsorpciji zračenja: {len(absorbance_measured_columns)}.")
    abs_measured_columns_without_co2 = remove_m_co2_spectar(absorbance_measured_columns)
    measured_columns = measured_columns + abs_measured_columns_without_co2

    df.sample(frac=1)
    x = df[measured_columns+spatial_columns]
    y = df[target_columns].values

    alpha = 0.1
    x_outliers, y_outliers, x_train, y_train = remove_rows_with_outlier(x, y, std_threshold=5)
    print(f"Outliers shape: {x_outliers.shape}")
    spatial_pca_model, x_spatial_components_train = transform_m_columns_pca(x_train[spatial_columns], 0.95)
    x_train = np.concatenate((x_spatial_components_train, x_train[measured_columns]), axis=1)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train, y_train)

    model = Ridge(alpha=alpha)
    model.fit(x_train, y_train)

    y_predicted = model.predict(x_train)
    score = np.abs(mean_columnwise_root_mse(y_train, y_predicted))
    print(f"Final model mean_columnwise_root_mse on training: {np.round(score, 2)}")
    filename = '../results/final_ridge.model'
    pickle.dump(model, open(filename, 'wb'))





