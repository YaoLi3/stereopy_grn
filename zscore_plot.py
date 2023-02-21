import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import stereo as st


def plot_reg_2d(zscore, coor, reg_name):
    # prepare plotting data
    sub_zscore = zscore[['Cell',reg_name]]
    # sort data points by zscore (low to high)
    zorder = np.argsort(sub_zscore[reg_name].values)
    # plot cell/bin dot, x y coor
    sc = plt.scatter(coor['x'][zorder], coor['y'][zorder], c=sub_zscore[reg_name][zorder], marker='.', edgecolors='none', cmap='plasma', lw=0)
    plt.box(False)
    plt.axis('off')
    plt.colorbar(sc,shrink=0.35)
    plt.savefig(f'{reg_name.split("(")[0]}_2d.png')
    plt.close()


def plot_multi_regs(zscore, coor, target_regs):
    pass


if __name__ == '__main__':
    auc_zscore = pd.read_csv('auc_zscore.csv')
    cell_coor = pd.read_csv('cell_coor.csv')
    #target_regs = ['Atf2(+)','Thra(+)', 'Wt1(+)']
    target_regs = list(auc_zscore.columns)
    for reg in target_regs:
        if '(+)' not in reg:
            continue
        plot_reg_2d(auc_zscore, cell_coor, reg)
        

