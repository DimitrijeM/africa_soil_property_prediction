from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import RegressorChain
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor

from app import remove_m_co2_spectar, remove_rows_with_outlier, transform_m_columns_pca, \
    mean_columnwise_root_mse_scorer

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
np.random.seed(1)
np.set_printoptions(formatter={'float_kind': lambda x: "%.3f" % x})


def defined_cross_val_score(x, y, estimators, n_folds=3, std_threshold=None):
    estimator_scores = {estimator_name: np.array([]) for estimator_name in estimators.keys()}

    for i in range(n_folds):
        data = pd.concat([x, y], axis=1, sort=False)
        data.sample(frac=1)
        x_shuffled, y_shuffled = data.iloc[:, :-5], data.iloc[:, -5:].values
        x_train, x_test, y_train, y_test = train_test_split(x_shuffled, y_shuffled, test_size=1/n_folds)

        if std_threshold is not None:
            x_outliers, y_outliers, x_train, y_train = remove_rows_with_outlier(x_train, y_train, std_threshold=5)
            print(f"Outliers shape: {x_outliers.shape}")
        spatial_pca_model, x_spatial_components_train = transform_m_columns_pca(x_train[spatial_columns], 0.95)
        x_train = np.concatenate((x_spatial_components_train, x_train[measured_columns]), axis=1)

        x_spatial_components_test = spatial_pca_model.transform(x_test[spatial_columns])
        x_test = np.concatenate((x_spatial_components_test, x_test[measured_columns]), axis=1)

        for estimator_name, estimator in estimators.items():
            pipeline = Pipeline([('scaler', MinMaxScaler()), (estimator_name, estimator)])
            pipeline.fit(x_train, y_train)
            y_predicted = pipeline.predict(x_test)

            score = np.abs(mean_columnwise_root_mse_scorer(pipeline, x_test, y_test))
            print(f"{i}. iter of {estimator_name}: {score}")
            estimator_scores[estimator_name] = np.append(estimator_scores[estimator_name], score)

    mean_estimator_scores = [{"model": e_name, "mean_score": e_score.mean(),  "std_score": e_score.std()} for e_name, e_score in estimator_scores.items()]
    scores_df = pd.DataFrame(mean_estimator_scores)
    return scores_df


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


    chain_order = [4, 3, 1, 2, 0]
    estimators = {
        "K-nn": KNeighborsRegressor(),
        "Ridge": Ridge(),
        "Lasso": Lasso(),
        "ElasticNet": ElasticNet(random_state=0),
        "MultiO/P AdaB": MultiOutputRegressor(AdaBoostRegressor(n_estimators=5)),
        "RegChain K-nn": RegressorChain(KNeighborsRegressor(), order=chain_order),
        "RandomForestRegressor": RandomForestRegressor(max_depth=4),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=5),
        "Extra trees": ExtraTreesRegressor(n_estimators=10),
        "MultiO/P GBR": MultiOutputRegressor(GradientBoostingRegressor(n_estimators=5))
    }
    scores_df = defined_cross_val_score(x, y, estimators, n_folds=3, std_threshold=None)
    scores_df = scores_df.sort_values(by='mean_score')
    print(scores_df)
    scores_df.to_csv(f'../results/own_crossval_full_{datetime.now().strftime("%Y-%m-%d-%H-%M")}.csv', index=False)


