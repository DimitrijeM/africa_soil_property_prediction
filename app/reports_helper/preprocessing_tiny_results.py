import pandas as pd


removed_outliers_evaluation_results = pd.read_csv('../results/evaluation_base_preprocessing_no_outliers_label_2020-06-28-20-31.csv')
best_models_eval = removed_outliers_evaluation_results.query('model == \'Ridge\' or model == \'K-nn\' or model=\'MultiO/P AdaB\'')
best_models_eval = best_models_eval.groupby('data_index')['score_mean'].mean().sort_values()
best_models_eval.to_csv('../../results/preprocessing_tiny_results.csv')
