import os
import sys
import argparse
import pandas as pd
from multiprocessing import cpu_count
from regulatory_network import InferenceRegulatoryNetwork as irn
from plot import PlotRegulatoryNetwork as prn
from pyscenic.rss import regulon_specificity_scores


def get_data(time_list, methods, top_genes):
    for time in time_list:
        data_fn = f'/dellfsqd2/ST_OCEAN/USER/liyao1/spatialGRN/DATA/{time}_pca.h5ad'
        data = irn.read_file(data_fn)
        for method in methods:
            base = f'/dellfsqd2/ST_OCEAN/USER/liyao1/spatialGRN/exp/fly3d/celltype_regulons/{time}_pca/'
            rss_fn = f'/dellfsqd2/ST_OCEAN/USER/liyao1/spatialGRN/exp/fly3d/celltype_regulons/{time}_pca/{method}_regulon_specificity_scores.txt'
            auc_mtx_fn = os.path.join(base,  f'{method}_auc.csv')
            auc_mtx = pd.read_csv(auc_mtx_fn, index_col = 0)
            data = data[auc_mtx.index, :]
            if os.path.isfile(rss_fn):
                rss_cellType = pd.read_csv(rss_fn, index_col = 0)
            else:
                anno = data.obs['annotation']
                anno = anno.loc[auc_mtx.index]
                rss_cellType = regulon_specificity_scores(auc_mtx, cell_type_series=anno)
                rss_cellType.to_csv(os.path.join(base, f'{method}_regulon_specificity_scores.txt'))
            #top_regs_dir = irn.get_top_regulons(data, 'annotation', rss_cellType, topn=10)
            #print(time, method)
            #print(top_regs_dir)
            #top_regs = prn.get_top_regulons(data, 'annotation', rss_cellType, topn=5)
            #total_top = total_top+top_regs
            #for celltype in top_regs_dir.keys():
            #    top_regs = top_regs_dir[celltype]
            for reg in total_top:
                if reg in auc_mtx.columns:
                    prn.plot_3d_reg(data, 'spatial', auc_mtx, reg_name=reg, fn=f'{reg.strip("(+)")}_{method}_{time}.png', vmin=0, vmax=7, alpha=0.3)
                    #prn.plot_2d_reg_h5ad(data, 'spatial', auc_mtx, reg_name=reg, fn=f'{reg.strip("(+)")}_{method}_{time}.png')


if __name__ == '__main__':
    methods = ['hotspot', 'scoexp']
    time_list = ['E14-16h', 'E16-18h', 'L1', 'L2', 'L3']
    #time_list=['E14-16h']
    #total_top = []
    #regulon_fn = sys.argv[1]
    #with open(regulon_fn,'r') as f:
    #    total_top = f.read().splitlines()
    #total_top=['hth(+)', 'GATAe(+)', 'CrebA(+)']
    total_top=['grh(+)']
    #with open('top_regulpns.txt','w') as f:
    #    f.writelines('\n'.join(list(set(total_top))))

