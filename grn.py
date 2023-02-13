# -*- coding: utf-8 -*-
import os
import time
import sys

import glob
import json
import pandas as pd
import loompy as lp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scanpy.pl import dotplot
from pyscenic.export import export2loom
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
from arboreto.utils import load_tf_names
from arboreto.algo import grnboost2
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.utils import load_motifs, modules_from_adjacencies
from pyscenic.prune import prune2df, df2regulons
from pyscenic.aucell import aucell

from stereo.io.reader import read_gef
from stereo.log_manager import LogManager
from stereo.plots.plot_base import PlotBase
from stereo.algorithm_base import AlgorithmBase, ErrorCode




def is_valid_exp_matrix(mtx:pd.DataFrame):
    '''check if the exp matrix is vaild for the grn pipline'''
    return (all(isinstance(idx, str) for idx in mtx.index) 
            and all(isinstance(idx, str) for idx in mtx.columns)
            and (mtx.index.nlevels == 1)
            and (mtx.columns.nlevels == 1))


def load_data(fn:str, bin_type='cell_bins'):
    logger.info('Loading expression data...')
    extension = os.path.splitext(fn)[1]
    if extension == '.csv':
        ex_mtx = pd.read_csv(fn)
        gene_names = list(ex_mtx.columns)
        return ex_mtx, gene_names
    elif extension == '.loom':
        pass
    elif extension == '.gef':
        data = read_gef(file_path=fn, bin_type=bin_type)
        genes = data.gene_names
        mtx = data.to_df()
        return mtx, genes


def _set_client(num_workers:int)->Client:
    local_cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)
    custom_client = Client(local_cluster)
    return custom_client


def grn_inference(ex_matrix, 
                    tf_names,
                    genes,
                    num_workers:int,
                    verbose=True,
                    fn:str='adj.csv')->pd.DataFrame:
    logger.info('GRN inferencing...')
    begin1 = time.time()
    custom_client = _set_client(num_workers)
    adjacencies = grnboost2(ex_matrix, 
                            tf_names=tf_names,
                            gene_names = genes,
                            verbose=True,
                            client_or_address=custom_client) 
    end1 = time.time()
    logger.info(f'GRN inference DONE in {(end1-begin1)//60} min {(end1-begin1)%60} sec')
    adjacencies.to_csv(fn, index=False)
    return adjacencies    


def regulons2csv(regulons:list, fn:str='regulons.csv'):
    '''
    Save regulons (df2regulons output) into a csv file.
    '''
    rdict={}
    for reg in regulons:
        targets = [ target for target in reg.gene2weight]
        rdict[reg.name] = targets

    #Optional: join list of target genes 
    for key in rdict.keys(): rdict[key]=";".join(rdict[key])

    #Write to csv file
    with open(fn,'w') as f:
        w = csv.writer(f)
        w.writerow(["Regulons","Target_genes"])
        w.writerows(rdict.items())


def name(fname:str)->str:
    return os.path.splitext(os.path.basename(fname))[0]


def load_database(DATABASES_GLOB:str) -> list: 
    logger.info('Loading ranked databases...')
    db_fnames = glob.glob(DATABASES_GLOB)
    dbs = [RankingDatabase(fname=fname, name=name(fname)) for fname in db_fnames]
    return dbs


def ctx_get_regulons(adjacencies:pd.DataFrame, 
                    ex_matrix:pd.DataFrame, 
                    dbs:list, 
                    MOTIF_ANNOTATIONS_FNAME:str, 
                    num_workers:int,
                    rho_mask_dropouts:bool=False,
                    is_prune:bool=True,
                    rgn:str='regulons.csv'):
    begin2 = time.time()
    modules = list(
        modules_from_adjacencies(
            adjacencies, 
            ex_matrix,
            rho_mask_dropouts=rho_mask_dropouts
            )
        )
    end2 = time.time()
    logger.info(f'Regulon Prediction DONE in {(end2-begin2)//60} min {(end2-begin2)%60} sec')
    logger.info(f'generated {len(modules)} modules')
    # 4.2 Create regulons from this table of enriched motifs.
    if is_prune:
        with ProgressBar():
           df = prune2df(dbs, modules, MOTIF_ANNOTATIONS_FNAME, num_workers=num_workers) 
        regulons = df2regulons(df)
        df.to_csv(rgn)
        return regulons
    else:
        pass


def auc_activity_level(ex_matrix,
                        regulons,
                        auc_thld,
                        num_workers,
                        fn='auc.csv')->pd.DataFrame:
    begin4 = time.time()
    auc_mtx = aucell(ex_matrix, regulons, auc_threshold=auc_thld, num_workers=num_workers)
    end4 = time.time()
    logger.info(f'Cellular Enrichment DONE in {(end4-begin4)//60} min {(end4-begin4)%60} sec')
    auc_mtx.to_csv(fn)
    return auc_mtx


def save2loom(auc_mtx, LOOM_FILE:str='output.loom'):
    #lf = lp.connect(LOOM_FILE, mode='r+', validate=False )
    #auc_mtx = pd.DataFrame(lf.ca.RegulonsAUC, index=lf.ca.CellID)
    #lf.close()
    LOOM_FILE = 'out.loom'
    export2loom(exp_mtx, regulons[0:100], annotations,
                LOOM_FILE,
                title = "grn",
                nomenclature = "MGI") #compress=True



if __name__ == '__main__':
    logger = LogManager(log_path='./SS200000135TL_D1.cellbin.log',level='debug').get_logger(name='Stereo')
    begin = time.time()
    RESOURCES_FOLDER="/dellfsqd2/ST_OCEAN/USER/liyao1/stereopy/resource"
    DATABASE_FOLDER = "/dellfsqd2/ST_OCEAN/USER/liyao1/stereopy/database/"
    DATABASES_GLOB = os.path.join(DATABASE_FOLDER,'mm10_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather') 
    MOTIF_ANNOTATIONS_FNAME = os.path.join(RESOURCES_FOLDER, 'motifs/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl')
    MM_TFS_FNAME = os.path.join(RESOURCES_FOLDER, 'tfs/test_mm_mgi_tfs.txt')
    SC_EXP_FNAME = os.path.join(RESOURCES_FOLDER, 'StereopyData/SS200000135TL_D1.cellbin.gef')


    # 0. Load StereoExpData file
    ex_matrix, genes = load_data(SC_EXP_FNAME)

    # 1. load TF list
    logger.info('Loading TF list...')
    tf_names = load_tf_names(MM_TFS_FNAME)


    # 2. load the ranking databases
    dbs = load_database(DATABASES_GLOB) 


    # 3. GRN inference
    adjacencies = pd.read_csv('adj.csv')

    unique_adj_genes = set(adjacencies["TF"]).union(set(adjacencies["target"])) - set(ex_matrix.columns)
    logger.info(f'find {len(unique_adj_genes)/len(set(ex_matrix.columns))} unique genes')

    # 4. Regulon prediction aka cisTarget from CLI
    regulons = ctx_get_regulons(adjacencies, ex_matrix, dbs, MOTIF_ANNOTATIONS_FNAME, num_workers=24)

    # 5: Cellular enrichment (aka AUCell) from CLI
    auc_mtx = auc_activity_level(ex_matrix, regulons, auc_thld=0.5, num_workers=24)
    
    # END
    end = time.time()
    logger.info(f'script finished in {(end-begin)//60} min {(end-begin)%60} sec')
   


