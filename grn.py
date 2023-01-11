# -*- coding: utf-8 -*-
import os
import glob
import pickle
import json
import pandas as pd
import numpy as np
from dask.diagnostics import ProgressBar
from arboreto.utils import load_tf_names
from arboreto.algo import grnboost2
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.utils import modules_from_adjacencies, load_motifs
from pyscenic.prune import prune2df, df2regulons
from pyscenic.aucell import aucell, derive_auc_threshold, create_rankings
import seaborn as sns
#----------------------------------
import scanpy as sc
from anndata import (read_h5ad, read_loom)

import argparse
#import signatures
import time
import logging
import sys
from stereo.io.reader import read_gef

#def load_signatures(fname: str) -> Sequence[Type[GeneSignature]]:
#    """
#    Load genes signatures from disk.
#    Supported file formats are GMT, DAT (pickled), YAML or CSV (enriched motifs).
#    :param fname: The name of the file that contains the signatures.
#    :return: A list of gene signatures.
#    """
#    extension = PurePath(fname).suffixes
#    if is_valid_suffix(extension, "ctx"):
#        # csv/tsv
#        return df2regulons(load_motifs(fname, sep=suffixes_to_separator(extension)))
#    elif is_valid_suffix(extension, "ctx_yaml"):
#        return load_from_yaml(fname)
#    elif ".gmt" in extension:
#        sep = guess_separator(fname)
#        return GeneSignature.from_gmt(fname, field_separator=sep, gene_separator=sep)
#    elif ".dat" in extension:
#        with openfile(fname, "rb") as f:
#            return pickle.load(f)
#    else:
#        raise ValueError('Unknown file format "{}".'.format(fname))

def is_valid_exp_matrix(mtx):
    return (all(isinstance(idx, str) for idx in mtx.index) 
            and all(isinstance(idx, str) for idx in mtx.columns)
            and (mtx.index.nlevels == 1)
            and (mtx.columns.nlevels == 1))


def heatmap():
    pass


def auc_map():
    pass


if __name__ == '__main__':
    begin = time.time()
    #DATA_FOLDER="~/tmp"
    RESOURCES_FOLDER="./resource"
    DATABASE_FOLDER = "./database/"
    DATABASES_GLOB = '/dellfsqd2/ST_OCEAN/USER/hankai/software/SpatialTranscript/scenic/cistarget_databases.plarian/planrian.regions_vs_motifs.rankings.feather'
    #DATABASES_GLOB = os.path.join(DATABASE_FOLDER, "hg19-tss-centered-10kb-10species.mc9nr.genes_vs_motifs.rankings.feather")
    #MOTIF_ANNOTATIONS_FNAME = os.path.join(RESOURCES_FOLDER, "motifs-v9-nr.hgnc-m0.001-o0.0.tbl")
    MOTIF_ANNOTATIONS_FNAME = '/dellfsqd2/ST_OCEAN/USER/hankai/software/SpatialTranscript/scenic/cistarget_databases.plarian/motifs-v9-nr.planarian-m0.001-o0.0.tbl'
    MM_TFS_FNAME = os.path.join(RESOURCES_FOLDER, 'TFs.txt')#'hs_hgnc_tfs.txt')#'mm_tfs.txt')
    #SC_EXP_FNAME = os.path.join(RESOURCES_FOLDER, 'WT_smes_cell_norm.csv')
    SC_EXP_FNAME = os.path.join(RESOURCES_FOLDER, sys.argv[1])
    
    
    #SC_EXP_FNAME = os.path.join(RESOURCES_FOLDER, "GSE60361_C1-3005-Expression.txt")
    #REGULONS_FNAME = os.path.join(DATA_FOLDER, "regulons.p")
    #MOTIFS_FNAME = os.path.join(DATA_FOLDER, "motifs.csv")


    # 0.should load in Stereopy data structure
    print('Loading expression data...')
    #ex_matrix = pd.read_csv(SC_EXP_FNAME)
    #adata = read_loom(SC_EXP_FNAME)
    #ex_matrix = adata.X
    #print(type(ex_matrix))
    #print(ex_matrix.head())
    data = read_gef(file_path=SC_EXP_FNAME, bin_type='cell_bins')
    print(data.shape)
    ex_matrix = data.exp_matrix
    genes = data.gene_names



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
    #adjacencies = grnboost2(ex_matrix, gene_names=list(ex_matrix.columns), verbose=True)
    adjacencies = grnboost2(ex_matrix, gene_names=list(genes), verbose=True)
    end1 = time.time()
    print(f'GRN inference DONE in {(end1-begin1)//60} min {(end1-begin1)%60} sec')
    adjacencies.to_csv('adj.csv', index=False)
    #adjacencies = pd.read_csv('adj.csv')
    unique_adj_genes = set(adjacencies["TF"]).union(set(adjacencies["target"])) - set(ex_matrix.columns)
    print(f'find {len(unique_adj_genes)/len(set(ex_matrix.columns))} unique genes')


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
        #pickle.dump(modules, f)
        #for m in modules:
            #f.writeline(m)
    
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
    #auc_mtx = aucell(
    #    ex_matrix,
    #    regulons,
    #    auc_threshold=0.5,
    #    noweights=False,
    #    normalize=False,
    #    seed=None,
    #    num_workers=24)
    auc_mtx = aucell(ex_matrix, regulons, auc_threshold=0.5, num_workers=24)
    end4 = time.time()
    print(f'Cellular Enrichment DONE in {(end4-begin4)//60} min {(end4-begin4)%60} sec')
    print(f'auc matrix is {type(auc_mtx)}')
    print(auc_mtx.head())
    auc_mtx.to_csv('auc.csv')
    
    # collect SCENIC AUCell output
    lf = lp.connect(f_pyscenic_output, mode='r+', validate=False )
    auc_mtx = pd.DataFrame(lf.ca.RegulonsAUC, index=lf.ca.CellID)
    lf.close()

    # END
    end = time.time()
    print(f'script finished in {(end-begin)//60} min {(end-begin)%60} sec')



