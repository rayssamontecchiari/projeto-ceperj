import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("TABELA_TRABALHADA.csv")

corr = dataset.corr()

# plot = sns.heatmap(corr, annot = True, fmt=".1f", linewidths=.6)
# plot

corr = dataset.corr()
sns.heatmap(corr, cmap = 'YlOrRd', linewidths=0.1)
plt.show()