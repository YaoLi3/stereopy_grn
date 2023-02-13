from pyscenic.rss import regulon_specificity_scores
from pyscenic.plotting import plot_rss
from pyscenic.export import add_scenic_metadata
from pyscenic.cli.utils import load_signatures
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import loompy as lp

from stereo.io import read_ann_h5ad, read_gef


if __name__ == '__main__':
    # scenic output
    lf = lp.connect('out.loom', mode='r', validate=False) # validate must set to False
    auc_mtx = pd.DataFrame(lf.ca.RegulonsAUC, index=lf.ca.CellID)

    data = read_ann_h5ad('/dellfsqd2/ST_OCEAN/USER/liyao1/stereopy/resource/StereopyData/Cellbin_deversion.h5ad')
    clusters_series = pd.read_csv('meta_mousebrain.csv',index_col=0).iloc[:, 0]


    # load the regulons from a file using the load_signatures function
    sig = load_signatures('regulons.csv')
    # TODO: adapt to StereoExpData
    #adata = add_scenic_metadata(adata, auc_mtx, sig)

    ### Regulon specificity scores (RSS) across predicted cell types
    ### Calculate RSS
    rss_cellType = regulon_specificity_scores(auc_mtx, clusters_series)
    #rss_cellType.to_csv('regulon_specificity_scores.txt')
    
    print(rss_cellType.shape)
    #sns.heatmap(rss_cellType)
    #plt.savefig('test_rss.png')

    func = lambda x: (x - x.mean()) / x.std(ddof=0)
    auc_mtx_Z = auc_mtx.transform(func, axis=0)
    auc_mtx_Z.to_csv('pyscenic_output.prune.zscore.csv')

    ### Select the top 5 regulons from each cell type
    cats = sorted(list(set(clusters_series)))
    topreg = []
    for i, c in enumerate(cats):
        topreg.extend(
                      list(rss_cellType.T[c].sort_values(ascending=False)[:5].index)
                     )
    topreg = list(set(topreg))


    colors = [
        '#d60000', '#e2afaf', '#018700', '#a17569', '#e6a500', '#004b00',
        '#6b004f', '#573b00', '#005659', '#5e7b87', '#0000dd', '#00acc6',
        '#bcb6ff', '#bf03b8', '#645472', '#790000', '#0774d8', '#729a7c',
        '#8287ff', '#ff7ed1', '#8e7b01', '#9e4b00', '#8eba00', '#a57bb8',
        '#5901a3', '#8c3bff', '#a03a52', '#a1c8c8', '#f2007b', '#ff7752',
        '#bac389', '#15e18c', '#60383b', '#546744', '#380000', '#e252ff',
        ]
    #colorsd = dict((f'c{i}', c) for i, c in enumerate(colors))
    #colormap = [colorsd[x] for x in clusters_series]




    sns.set(font_scale=1.2)
    g = sns.clustermap(auc_mtx_Z[topreg], annot=False,  square=False,  linecolor='gray',yticklabels=True, xticklabels=True, vmin=-2, vmax=6, cmap="YlGnBu", figsize=(21,16) )
    g.cax.set_visible(True)
    g.ax_heatmap.set_ylabel('')
    g.ax_heatmap.set_xlabel('')
    plt.savefig("clusters-heatmap-top5.png")


