import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

plt.rcParams["figure.figsize"] = (12,6)

df = pd.read_csv('../../data/training.csv').set_index('PIDN')
target_columns = ['Ca', "P", "pH", "SOC", "Sand"]
measured_columns = ['Depth']
spatial_columns = [x for x in list(df.columns) if x not in (target_columns+measured_columns) and not x.startswith('m')]
spatial_df = df[spatial_columns]
pca = PCA(n_components=0.99).fit(spatial_df.values)

fig, ax = plt.subplots()
xi = np.arange(1, 10, step=1)
y = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.3, 1.1)
plt.plot(xi, y, marker='o', linestyle='--', color='b')

plt.xlabel('Broj komponenti')
plt.xticks(np.arange(0, 11, step=1))
plt.ylabel('Kumulativna varijacija (%)')
plt.title('Broj komponenti potreban da objasni varijaciju')

plt.axhline(y=0.95, color='r', linestyle='-')
plt.text(3, 1, '95% prag', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.savefig('../../img/spatial_pca_graph.png')
