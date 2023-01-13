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
from pyscenic.export import export2loom
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
from arboreto.utils import load_tf_names
from arboreto.algo import grnboost2
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.utils import modules_from_adjacencies, load_motifs
from pyscenic.prune import prune2df, df2regulons
from pyscenic.aucell import aucell, derive_auc_threshold, create_rankings
from pyscenic.prune import _prepare_client

from stereo.io.reader import read_gef
#from st_pyscenic.algo import grnboost2


def is_valid_exp_matrix(mtx):
    return (all(isinstance(idx, str) for idx in mtx.index) 
            and all(isinstance(idx, str) for idx in mtx.columns)
            and (mtx.index.nlevels == 1)
            and (mtx.columns.nlevels == 1))




if __name__ == '__main__':
    begin = time.time()
    RESOURCES_FOLDER="./resource"
    DATABASE_FOLDER = "./database/"
    DATABASES_GLOB = '/dellfsqd2/ST_OCEAN/USER/hankai/software/SpatialTranscript/scenic/cistarget_databases.plarian/planrian.regions_vs_motifs.rankings.feather'
    MOTIF_ANNOTATIONS_FNAME = '/dellfsqd2/ST_OCEAN/USER/hankai/software/SpatialTranscript/scenic/cistarget_databases.plarian/motifs-v9-nr.planarian-m0.001-o0.0.tbl'
    MM_TFS_FNAME = os.path.join(RESOURCES_FOLDER, 'TFs.txt')#'hs_hgnc_tfs.txt')#'mm_tfs.txt')
    SC_EXP_FNAME = 'resource/WT_smes_cell_norm.csv'
    #SC_EXP_FNAME = sys.argv[1]
    #MM_TFS_FNAME = sys.argv[2]


    # 0.should load in Stereopy data structure
    print('Loading expression data...')
    #data = read_gef(file_path=SC_EXP_FNAME, bin_type='cell_bins')
    #print(data.shape)
    #ex_matrix = data.exp_matrix
    #genes = data.gene_names
    ex_matrix = pd.read_csv(SC_EXP_FNAME)
    genes = list(ex_matrix.columns)


    # 1. load TF list
    print('Loading TF list...')
    tf_names = load_tf_names(MM_TFS_FNAME)
    print(len(tf_names))


    # 2. load the ranking databases
    print('Loading ranked databases...')
    db_fnames = glob.glob(DATABASES_GLOB)
    def name(fname):
        return os.path.splitext(os.path.basename(fname))[0]
    dbs = [RankingDatabase(fname=fname, name=name(fname)) for fname in db_fnames]
    print(dbs)


    # 3. GRN inference
    print('GRN inferencing...')
    begin1 = time.time()
    #adjacencies = grnboost2(ex_matrix, gene_names=list(genes), verbose=True)
   
    # method 1: change _prepare_client code that is called by grnboost2
    #adjacencies = grnboost2(ex_matrix, tf_names=tf_names, verbose=True)
    
    # method 2: import _prepare_client from prune module, create customed client
    # 'local_host'?
    #client, shutdown_callback = _prepare_client('local_host', num_workers=12)
    #network = grnboost2(expression_data=ex_matrix, tf_names=tf_names, verbose=True, client_or_address=client)
    
    # method 3: impoer LocalCluster, Client from dask, create custom Client
    local_cluster = LocalCluster(n_workers=24, threads_per_worker=1)
    custom_client = Client(local_cluster)
    adjacencies = grnboost2(ex_matrix, 
                            tf_names=tf_names, 
                            verbose=True,
                            client_or_address=custom_client) 
    end1 = time.time()
    print(f'GRN inference DONE in {(end1-begin1)//60} min {(end1-begin1)%60} sec')
    adjacencies.to_csv('adj.csv', index=False)
    unique_adj_genes = set(adjacencies["TF"]).union(set(adjacencies["target"])) - set(ex_matrix.columns)
    print(f'find {len(unique_adj_genes)/len(set(ex_matrix.columns))} unique genes')

    '''
    #Adjacencies based on GRN. Here we see an 'importance' score for each target gene and Transcription factor relationship.
    #Later after pruning, we get regulons. In this output we have weights for each gene for each MOTIF of each Transcription factor.

    # 4. Regulon prediction aka cisTarget from CLI
    out_fname = 'reg.json'
    begin2 = time.time()
    modules = list(
        modules_from_adjacencies(
            adjacencies, 
            ex_matrix,
            rho_mask_dropouts=False
            )
        )
    end2 = time.time()
    print(f'Regulon Prediction DONE in {(end2-begin2)//60} min {(end2-begin2)%60} sec')
    print(f'generated {len(modules)} modules')
    print(type(modules))
    print(type(modules[0]))

    with open(out_fname, "wb", encoding='utf-8') as f:
        json.dump(modules, f, indent=4)
    
    # 4.2 Create regulons from this table of enriched motifs.
    with ProgressBar():
        df = prune2df(dbs, modules, MOTIF_ANNOTATIONS_FNAME, num_workers=24)
    regulons = df2regulons(df)
    print(f'regulons are {type(regulons)}')
    print(f'df is {type(df)}')
    df.to_csv('motif.csv')
    with open('regulons.json', 'w', encoding='utf-8') as f:
        json.dump(regulons, f, indent=4)


    
    # 5: Cellular enrichment (aka AUCell) from CLI
    begin4 = time.time()
    auc_mtx = aucell(ex_matrix, regulons, auc_threshold=0.5, num_workers=24)
    end4 = time.time()
    print(f'Cellular Enrichment DONE in {(end4-begin4)//60} min {(end4-begin4)%60} sec')
OM_FILE
    print(f'auc matrix is {type(auc_mtx)}')
    print(auc_mtx.head())
    auc_mtx.to_csv('auc.csv')
    # collect SCENIC AUCell output
    lf = lp.connect(f_pyscenic_output, mode='r+', validate=False )
    auc_mtx = pd.DataFrame(lf.ca.RegulonsAUC, index=lf.ca.CellID)
    lf.close()

    # save results in a loom file
    LOOM_FILE = 'out.loom'
    export2loom(exp_mtx, regulons[0:100], annotations,
                LOOM_FILE,
                title = "xx",
                nomenclature = "MGI") #compress=True
    
    '''
    # END
    end = time.time()
    print(f'script finished in {(end-begin)//60} min {(end-begin)%60} sec')



