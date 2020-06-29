import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import MinMaxScaler

from app import remove_m_co2_spectar, remove_rows_with_outlier, transform_m_columns_pca, scale_x

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.random.seed(1)
np.set_printoptions(formatter={'float_kind': lambda x: "%.3f" % x})


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

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

    x_outliers, y_outliers, x, y = remove_rows_with_outlier(x_train, y_train, std_threshold=5)
    print(f"Outliers shape: {x_outliers.shape}")
    spatial_pca_model, x_spatial_components_train = transform_m_columns_pca(x_train[spatial_columns], 0.95)
    x_train = np.concatenate((x_spatial_components_train, x_train[measured_columns]), axis=1)

    x_spatial_components_test = spatial_pca_model.transform(x_test[spatial_columns])
    x_test = np.concatenate((x_spatial_components_test, x_test[measured_columns]), axis=1)

    scaler, x_train = scale_x(x_train, MinMaxScaler())
    lasso = Lasso()
    lasso.fit(x_train, y_train)
    print(lasso.coef_)
