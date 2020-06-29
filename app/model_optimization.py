import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler

from app import remove_m_co2_spectar, remove_rows_with_outlier, transform_m_columns_pca

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.random.seed(1)
np.set_printoptions(formatter={'float_kind': lambda x: "%.3f" % x})


def mean_columnwise_root_mse(y, y_pred):
    return np.sum((y - y_pred)**2/y.shape[0])/y.shape[1]


def grid_cv_ridge(x, y, alphas, n_folds=3):
    estimator_scores = np.array([])
    estimator_stds = np.array([])
    for alpha in alphas:
        scores = np.array([])
        for i in range(n_folds):
            data = pd.concat([x, y], axis=1, sort=False)
            data.sample(frac=1)
            x_shuffled, y_shuffled = data.iloc[:, :-5], data.iloc[:, -5:].values
            x_train, x_test, y_train, y_test = train_test_split(x_shuffled, y_shuffled, test_size=0.2)

            x_outliers, y_outliers, x_train, y_train = remove_rows_with_outlier(x_train, y_train, std_threshold=5)
            print(f"Outliers shape: {x_outliers.shape}")
            spatial_pca_model, x_spatial_components_train = transform_m_columns_pca(x_train[spatial_columns], 0.95)
            x_train = np.concatenate((x_spatial_components_train, x_train[measured_columns]), axis=1)

            x_spatial_components_test = spatial_pca_model.transform(x_test[spatial_columns])
            x_test = np.concatenate((x_spatial_components_test, x_test[measured_columns]), axis=1)

            scaler = MinMaxScaler()
            x_train = scaler.fit_transform(x_train, y_train)

            model = Ridge(alpha=alpha)
            model.fit(x_train, y_train)

            x_test = scaler.transform(x_test)
            y_predicted = model.predict(x_test)
            score = np.abs(mean_columnwise_root_mse(y_test, y_predicted))
            print(f"{i}. iter of {str(alpha)}: {score}")
            scores = np.append(scores, score)

        estimator_scores = np.append(estimator_scores, np.mean(scores))
        estimator_stds = np.append(estimator_stds, np.std(scores))

    best_i = np.argmin(estimator_scores)
    alpha = alphas[best_i]

    print(f"Best score with alpha={alpha}: {np.round(estimator_scores[best_i],2)} +- {np.round(estimator_stds[best_i],2)}")
    evaluation_df = pd.DataFrame([alphas, estimator_scores, estimator_stds]).T
    evaluation_df.columns = ['alpha', 'mean_columnwise_root_mse', 'std']
    evaluation_df = evaluation_df.sort_values(by='mean_columnwise_root_mse')
    evaluation_df = evaluation_df.round(2)
    evaluation_df.to_csv('../results/ridge_grid_apha_scores.csv')
    return evaluation_df


def grid_cv_knn(x, y, ks, weights, n_folds=3):
    possibles = {}
    for w in weights:
        possible_w = {f'{w}_{str(k)}': (w, k) for k in ks}
        possibles = {**possibles, **possible_w}

    estimator_scores = np.array([])
    estimator_stds = np.array([])
    for possible_key, possible_values in possibles.items():
        scores = np.array([])
        for i in range(n_folds):
            data = pd.concat([x, y], axis=1, sort=False)
            data.sample(frac=1)
            x_shuffled, y_shuffled = data.iloc[:, :-5], data.iloc[:, -5:].values
            x_train, x_test, y_train, y_test = train_test_split(x_shuffled, y_shuffled, test_size=0.2)

            x_outliers, y_outliers, x_train, y_train = remove_rows_with_outlier(x_train, y_train, std_threshold=5)
            print(f"Outliers shape: {x_outliers.shape}")
            spatial_pca_model, x_spatial_components_train = transform_m_columns_pca(x_train[spatial_columns], 0.95)
            x_train = np.concatenate((x_spatial_components_train, x_train[measured_columns]), axis=1)

            x_spatial_components_test = spatial_pca_model.transform(x_test[spatial_columns])
            x_test = np.concatenate((x_spatial_components_test, x_test[measured_columns]), axis=1)

            scaler = MinMaxScaler()
            x_train = scaler.fit_transform(x_train, y_train)

            w, k = possible_values
            model = KNeighborsRegressor(n_neighbors=k, weights=w)
            model.fit(x_train, y_train)

            x_test = scaler.transform(x_test)
            y_predicted = model.predict(x_test)
            score = np.abs(mean_columnwise_root_mse(y_test, y_predicted))
            print(f"{i}. iter of {possible_key}: {score}")
            scores = np.append(scores, score)

        estimator_scores = np.append(estimator_scores, np.mean(scores))
        estimator_stds = np.append(estimator_stds, np.std(scores))

    best_i = np.argmin(estimator_scores)
    best_possible = list(possibles.items())[best_i]
    print(f"Best score with {best_possible}: {np.round(estimator_scores[best_i],2)} +- {np.round(estimator_stds[best_i],2)}")
    evaluation_df = pd.DataFrame(list(possibles.values()))
    evaluation_df.columns = ['weights', 'k']
    evaluation_df['mean_columnwise_root_mse'] = estimator_scores
    evaluation_df['std'] = estimator_stds
    evaluation_df = evaluation_df.round(2)
    evaluation_df = evaluation_df.sort_values(by='mean_columnwise_root_mse')
    evaluation_df.to_csv('../results/knn_grid_apha_scores.csv')
    return evaluation_df


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
    y = df[target_columns]
    alphas = np.linspace(0.1, 2, 20)
    # print(f"Possible aplhas: {alphas}")
    grid_cv_ridge(x, y, alphas, n_folds=5)

    ks = list(range(1, 10))
    weights = ['uniform', 'distance']
    grid_cv_knn(x, y, ks, weights, n_folds=5)





