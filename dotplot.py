from scanpy.pl import dotplot
import pandas as pd
from stereo.io import read_gef
import sys
import matplotlib.pyplot as plt
import scanpy as sc


def _cal_percent_df(exp_matrix, cluster_meta, regulon, ct, cutoff=0):
    """
    Expression percent
    cell numbers
    """
    cells = cluster_meta['cluster' == ct]['cell']
    ct_exp = exp_matrix.iloc(cells)
    g_ct_exp = ct_exp[regulon]
    regulon_cell_num = g_ct_exp[g_ct_exp>cutoff].count()
    total_cell_num = 0
    return regulon_cell_num/total_cell_num


def _cal_exp_df(exp_matrix, cluster_meta, regulon, ct):
    cells = cluster_meta['cluster' == ct]['cell']
    ct_exp = exp_matrix.iloc(cells)
    g_ct_exp = ct_exp[regulon]
    return np.mean(g_ct_exp)
    

def subset_data(data, regulon, celltypes):
    pass
    
    
def dotplot(StereoExpData, **kwargs):
    '''
    Intuitive way of visualizing how feature expression changes across different
    identity classes (clusters). The size of the dot encodes the percentage of
    cells within a class, while the color encodes the AverageExpression level
    across all cells within a class (blue is high).
    
    @param features Input vector of features, or named list of feature vectors
    if feature-grouped panels are desired
    '''
    pass


if __name__ == '__main__':
    fn = sys.argv[1]

    data = read_gef(fn, bin_type='cell_bins')
    exp_mtx = data.to_df()
    smtx = data.exp_matrix
    print(type(smtx))

    auc_mtx = pd.read_csv('auc.csv',index_col=0)
    
    clusters = pd.read_csv(sys.argv[2])
    print(clusters.head())

    #plt.figure(figsize=(6,6))
    #dotplot(data, var_names=data.gene_names, groupby='Cell')
    #plt.savefig('dotplot.png')


    '''
    adata = sc.datasets.pbmc68k_reduced()
    markers = ['C1QA', 'PSAP', 'CD79A', 'CD79B', 'CST3', 'LYZ']
    sc.pl.DotPlot(adata, markers, groupby='bulk_labels')
    plt.savefig('dotplot.png')
    plt.close()

    print(adata)
    '''








