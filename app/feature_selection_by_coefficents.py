import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

from app import remove_m_co2_spectar, remove_rows_with_outlier, transform_m_columns_pca, \
    mean_columnwise_root_mse_scorer

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.random.seed(1)
np.set_printoptions(formatter={'float_kind': lambda x: "%.3f" % x})


def weighted_distance(x1, x2):
    return np.sum(np.sum(np.abs((x1-x2)))*distance_weights)


if __name__ == '__main__':
    df = pd.read_csv('../data/training.csv').set_index('PIDN')
    df.Depth = df.Depth.replace({'Subsoil': -1, 'Topsoil': 1})
    df.sample(frac=1)
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

    # choose columns
    x = df[measured_columns+spatial_columns]
    y = df[target_columns].values

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    x_outliers, y_outliers, x, y = remove_rows_with_outlier(x_train, y_train, std_threshold=5)
    print(f"Outliers shape: {x_outliers.shape}")

    spatial_pca_model, x_spatial_components_train = transform_m_columns_pca(x_train[spatial_columns], 0.95)
    x_train = np.concatenate((x_spatial_components_train, x_train[measured_columns]), axis=1)

    x_spatial_components_test = spatial_pca_model.transform(x_test[spatial_columns])
    x_test = np.concatenate((x_spatial_components_test, x_test[measured_columns]), axis=1)

    scaler = MinMaxScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    ridge = Ridge()
    ridge.fit(x_train, y_train)
    distance_weights = np.abs(ridge.coef_).sum(axis=0)

    print(mean_columnwise_root_mse_scorer(ridge, x_test, y_test))

    nbrs = KNeighborsRegressor(n_neighbors=4, algorithm='ball_tree', metric=weighted_distance)
    nbrs.fit(x_train, y_train)
    print(mean_columnwise_root_mse_scorer(nbrs, x_test, y_test))

