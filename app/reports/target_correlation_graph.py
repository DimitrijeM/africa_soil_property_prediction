import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('../data/training.csv').set_index('PIDN')
target_columns = ['Ca', "P", "pH", "SOC", "Sand"]
corr_matrix = df[target_columns].corr(method='pearson')
sns.heatmap(data=corr_matrix, annot=True, fmt=".1f")
plt.savefig('../../img/target_correlation.png')
