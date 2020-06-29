import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import RegressorChain
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.feature_selection import VarianceThreshold

from sklearn.tree import DecisionTreeRegressor

from sklearn.multioutput import MultiOutputRegressor
from datetime import datetime
from scipy import stats

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.random.seed(1)
np.set_printoptions(formatter={'float_kind': lambda x: "%.3f" % x})


def remove_m_co2_spectar(absorbance_measured_columns):
    absorbance_measured_columns_trans = [m for m in absorbance_measured_columns
                                         if (float(m.replace('m', '')) >= 2379.76
                                             or float(m.replace('m', '')) <= 2352.76)]
    print(f"Number of m features: {len(absorbance_measured_columns_trans)}")
    return absorbance_measured_columns_trans


def scale_x(x, scaler=MinMaxScaler()):
    x_trans = scaler.fit_transform(x)
    return scaler, x_trans


def remove_rows_with_outlier(x, y, std_threshold=5):
    is_outlier = (np.abs(stats.zscore(x)) > std_threshold).any(axis=1)
    x_no_outliers = x[np.logical_not(is_outlier)]
    y_no_outliers = y[np.logical_not(is_outlier)]
    x_outliers = x[is_outlier]
    y_outliers = y[is_outlier]
    return x_outliers, y_outliers, x_no_outliers, y_no_outliers


def reduce_with_low_variance(x, variance_threshold=0.05):
    x_trans = VarianceThreshold(variance_threshold).fit_transform(x)
    print(f"Variance threshold {variance_threshold}: {x.shape[1]-x_trans.shape[1]} features removed, returning {x_trans.shape[1]} features.")
    return x_trans


def transform_m_columns_pca(x, variance_threshold=0.95):
    print(f"Number of source features: {x.shape[1]}")
    pca_model = PCA(n_components=variance_threshold)
    x_trans = pca_model.fit_transform(x)
    print(f"Number of PCA components: {x_trans.shape[1]}")
    return pca_model, x_trans


def append_scaled_data(data_list):
    data_list_size = len(data_list)
    for i in range(data_list_size):
        data_items = data_list[i]
        data_list.append({"name": f"minmax_scaled_{data_items['name']}",
                          "x": scale_x(data_items["x"], MinMaxScaler())[1],
                          "y": data_items["y"]})
        data_list.append({"name": f"standard_scaled_{data_items['name']}",
                          "x": scale_x(data_items["x"], StandardScaler())[1],
                          "y": data_items["y"]})
    return data_list


def cross_validate_defined_models(data, models, n_folds=3, scoring_func='neg_mean_squared_error'):
    all_scores = pd.DataFrame()
    for model_name, model in models.items():
        scores = np.abs(cross_val_score(model, data['x'], data['y'], cv=n_folds, scoring=mean_columnwise_root_mse_scorer))
        print(f"Model {model_name}: {np.mean(scores)} +- {np.std(scores)} on {n_folds} folds.")
        all_scores = all_scores.append({
            "data_index": data['name'],
            "model": model_name,
            "n_folds": n_folds,
            "score": scoring_func,
            "score_mean": np.mean(scores),
            "score_std": np.std(scores)
        }, ignore_index=True)
    return all_scores


def evaluate_preprocessed_data(data_list, models, n_folds=3):
    evaluation_results = pd.DataFrame()
    for data_item in data_list:
        print(f"Evaluate on data: {data_item['name']}")
        evaluation_results = evaluation_results.append(cross_validate_defined_models(data_item, models, n_folds))
    return evaluation_results


def mean_columnwise_root_mse_scorer(estimator, x, y):
    y_pred = estimator.predict(x)
    return np.sum((y - y_pred)**2/y.shape[0])/y.shape[1]


if __name__ == '__main__':
    df = pd.read_csv('../data/training.csv').set_index('PIDN')
    df.Depth = df.Depth.replace({'Subsoil': -1, 'Topsoil': 1})
    df.sample(frac=1)
    print(df.shape)

    target_columns = ['Ca', "P", "pH", "SOC", "Sand"]
    measured_columns = ['Depth']
    spatial_columns = [x for x in list(df.columns) if x not in (target_columns+measured_columns) and not x.startswith('m')]
    absorbance_measured_columns = [x for x in list(df.columns) if x not in target_columns and x.startswith('m')]
    print(f"Ciljni atributi: {len(target_columns)}. Prostorni atributi: {len(spatial_columns)}. "
          f"Izmereni atributi: {len(measured_columns)}. "
          f"Atributi o izmerenoj apsorpciji zračenja: {len(absorbance_measured_columns)}.")

    # remove_rows_with_outlier
    # df = pd.DataFrame(remove_rows_with_outlier(df.values, 5)[1], columns=df.columns)
    # print(df.shape)

    full_x = df[measured_columns+spatial_columns+absorbance_measured_columns].values
    abs_measured_columns_without_co2 = remove_m_co2_spectar(absorbance_measured_columns)
    full_x_wo = df[measured_columns+abs_measured_columns_without_co2+spatial_columns]
    spatial_x = df[spatial_columns]
    measured_x = df[measured_columns+abs_measured_columns_without_co2]
    y = df[target_columns].values

    data_list = list()
    spatial_x_components = transform_m_columns_pca(scale_x(spatial_x)[1], 0.95)[1]
    measured_x_components = transform_m_columns_pca(scale_x(measured_x)[1], 0.95)[1]
    full_x_components = transform_m_columns_pca(scale_x(full_x_wo)[1], 0.95)[1]

    x_trasformed_high_var = reduce_with_low_variance(full_x_wo, 0.05)

    data_list.append({"name": "full", "x": full_x, "y": y})
    data_list.append({"name": "full_wo_co2", "x": full_x_wo, "y": y})
    data_list.append({"name": "high_var_x", "x": x_trasformed_high_var, "y": y})
    data_list.append({"name": "full_components", "x": full_x_components, "y": y})
    data_list.append({"name": "spatial_comp+m", "x": np.concatenate((spatial_x_components, measured_x), axis=1), "y": y})
    data_list.append({"name": "spatial+m_comp", "x": np.concatenate((spatial_x, measured_x_components), axis=1), "y": y})
    data_list = append_scaled_data(data_list)

    chain_order = [4, 3, 1, 2, 0]
    estimators = {
        "K-nn": KNeighborsRegressor(),
        # "Linear regression": LinearRegression(),
        "Ridge": Ridge(),
        # "Lasso": Lasso(),
        "ElasticNet": ElasticNet(random_state=0),
        # "RandomForestRegressor": RandomForestRegressor(max_depth=4),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=5),
        # "Extra trees": ExtraTreesRegressor(n_estimators=10, random_state=0),
        # "MultiO/P GBR": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=5)),
        "MultiO/P AdaB": MultiOutputRegressor(AdaBoostRegressor(n_estimators=5)),
        "RegChain K-nn": RegressorChain(KNeighborsRegressor(), order=chain_order)
    }

    evaluation_results = evaluate_preprocessed_data(data_list, estimators, n_folds=3)
    evaluation_results = evaluation_results.sort_values(by='score_mean')
    print(evaluation_results)
    evaluation_results.to_csv(f'../results/evaluation_base_preprocessing_no_outliers_label_{datetime.now().strftime("%Y-%m-%d-%H-%M")}.csv', index=False)





