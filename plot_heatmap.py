import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    auc_mtx = pd.read_csv('auc.csv',index_col=0)

    sns.clustermap(auc_mtx,figsize=(8,8))
    plt.savefig('auc_heatmap.png')
    plt.close()


