# -*- coding: utf-8 -*-
import os
import glob
import pickle
import pandas as pd
import numpy as np
from dask.diagnostics import ProgressBar
from arboreto.utils import load_tf_names
from arboreto.algo import grnboost2
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.utils import modules_from_adjacencies, load_motifs
from pyscenic.prune import prune2df, df2regulons
from pyscenic.aucell import aucell, , derive_auc_threshold, create_rankings
import seaborn as sns
#----------------------------------
import scanpy as sc
from anndata import (read_h5ad, read_loom)

import argparse
import signaturess
import time
import logging


def load_signatures(fname: str) -> Sequence[Type[GeneSignature]]:
    """
    Load genes signatures from disk.
    Supported file formats are GMT, DAT (pickled), YAML or CSV (enriched motifs).
    :param fname: The name of the file that contains the signatures.
    :return: A list of gene signatures.
    """
    extension = PurePath(fname).suffixes
    if is_valid_suffix(extension, "ctx"):
        # csv/tsv
        return df2regulons(load_motifs(fname, sep=suffixes_to_separator(extension)))
    elif is_valid_suffix(extension, "ctx_yaml"):
        return load_from_yaml(fname)
    elif ".gmt" in extension:
        sep = guess_separator(fname)
        return GeneSignature.from_gmt(fname, field_separator=sep, gene_separator=sep)
    elif ".dat" in extension:
        with openfile(fname, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError('Unknown file format "{}".'.format(fname))


if __name__ == '__main__':
    begin = time.time()
    #DATA_FOLDER="~/tmp"
    RESOURCES_FOLDER="./resource"
    DATABASE_FOLDER = "./database/"
    #SCHEDULER="123.122.8.24:8786"
    DATABASES_GLOB = os.path.join(DATABASE_FOLDER, "hg19-tss-centered-10kb-10species.mc9nr.genes_vs_motifs.rankings.feather")
    MOTIF_ANNOTATIONS_FNAME = os.path.join(RESOURCES_FOLDER, "motifs-v9-nr.hgnc-m0.001-o0.0.tbl")
    MM_TFS_FNAME = os.path.join(RESOURCES_FOLDER, 'TFs.txt')#'hs_hgnc_tfs.txt')#'mm_tfs.txt')
    SC_EXP_FNAME = os.path.join(RESOURCES_FOLDER, 'WT_smes_cell_norm.csv')
    #SC_EXP_FNAME = os.path.join(RESOURCES_FOLDER, "GSE60361_C1-3005-Expression.txt")
    #REGULONS_FNAME = os.path.join(DATA_FOLDER, "regulons.p")
    #MOTIFS_FNAME = os.path.join(DATA_FOLDER, "motifs.csv")


    #f_loom_path_scenic = sys.argv[1]
    # 0.should load in Stereopy data structure
    #ex_matrix = pd.read_csv(SC_EXP_FNAME, sep='\t', header=0, index_col=0).T
    ex_matrix = pd.read_csv(SC_EXP_FNAME)
    print(type(ex_matrix))
    print(ex_matrix.head())


    # import data
    #adata = sc.read_loom(fn)
    #adata = read_loom(f_loom_path_scenic)
    #print(adata)
    # extract expression matric from loom data structure
    #ex_matrix = adata.X

    #nCountsPerGene = np.sum(adata.X, axis=0)
    #nCellsPerGene = np.sum(adata.X>0, axis=0)
    ## Show info
    #print("Number of counts (in the dataset units) per gene:", nCountsPerGene.min(), " - " ,nCountsPerGene.max())
    #print("Number of cells in which each gene is detected:", nCellsPerGene.min(), " - " ,nCellsPerGene.max())
    #nCells=adata.X.shape[0]

    ## pySCENIC thresholds
    #minCountsPerGene=3*.01*nCells # 3 counts in 1% of cells
    #print("minCountsPerGene: ", minCountsPerGene)

    #minSamples=.01*nCells # 1% of cells
    #print("minSamples: ", minSamples)
    ## simply compute the number of genes per cell (computers 'n_genes' column)
    #sc.pp.filter_cells(adata, min_genes=0)
    ## mito and genes/counts cuts
    #mito_genes = adata.var_names.str.startswith('MT-')
    ## for each cell compute fraction of counts in mito genes vs. all genes
    #adata.obs['percent_mito'] = np.sum(
    #    adata[:, mito_genes].X, axis=1).A1 / np.sum(adata.X, axis=1).A1
    ## add the total counts per cell as observations-annotation to adata
    #adata.obs['n_counts'] = adata.X.sum(axis=1).A1


    # 1. load TF list
    tf_names = load_tf_names(MM_TFS_FNAME)
    print(type(tf_names))
    print(len(tf_names))


    # 2. load the ranking databases
    db_fnames = glob.glob(DATABASES_GLOB)
    def name(fname):
        return os.path.splitext(os.path.basename(fname))[0]

    dbs = [RankingDatabase(fname=fname, name=name(fname)) for fname in db_fnames]
    print(dbs)

    # ranking databases
    #f_db_glob = "/ddn1/vol1/staging/leuven/res_00001/databases/cistarget/databases/homo_sapiens/hg38/refseq_r80/mc9nr/gene_based/*feather"
    #f_db_names = ' '.join( glob.glob(f_db_glob) )


    # genes (n_cells x n_genes).
    #test_df = create_rankings(ex_matrix, seend=None)
    # aucell内部会call create_rankings这个函数，所以不需要自己调用

    # motif databases
    #f_motif_path = "/ddn1/vol1/staging/leuven/res_00001/databases/cistarget/motif2tf/motifs-v9-nr.hgnc-m0.001-o0.0.tbl"

    # 3. GRN inference
    #pyscenic grn {f_loom_path_scenic} {MM_TFS_FNAME} -o adj.csv --num_workers 20
    
    # ex_matrix, f_loom_path_scenic
    adjacencies = grnboost2(ex_matrix, gene_names=list(ex_matrix.columns), verbose=True)
    adjacencies.to_csv('adj.csv', index=False)
    #adjacencies = pd.read_csv("adj.tsv", index_col=False, sep='\t')
    print(adjacencies.head())
    
    unique_adj_genes = set(adjacencies["TF"]).union(set(adjacencies["target"])) - set(ex_matrix.columns)
    print(f'find {len(unique_adj_genes)/len(set(ex_matrix.columns))} unique genes')

    #Adjacencies based on GRN. Here we see an 'importance' score for each target gene and Transcription factor relationship.
    #Later after pruning, we get regulons. In this output we have weights for each gene for each MOTIF of each Transcription factor.

    # 4. Regulon prediction aka cisTarget from CLI
    #pyscenic ctx adj.tsv \
    #    {dbs} \
    #    --annotations_fname {MOTIF_ANNOTATIONS_FNAME} \
    #    --expression_mtx_fname {f_loom_path_scenic} \
    #    --output reg.csv \
    #    --mask_dropouts \
    #    --num_workers 20
    out_fname = 'reg.pickle' 
    modules = list(
        modules_from_adjacencies(
            adjacencies, 
            ex_matrix,
            rho_mask_dropouts=False
            )
        )
    print(f'generated {len(modules)} modules')
    #with open(out_fname, "wb") as f:
    #    pickle.dump(modules, f) 
    # pick out modules for Atoh1
    tf = 'SMESG000034284.1'
    tf_mods = [ x for x in modules if x.transcription_factor==tf ]
    print(tf_mods)

    for i,mod in enumerate( tf_mods ):
        print( f'{tf} module {str(i)}: {len(mod.genes)} genes' )
        print( f'{tf} regulon: {len(regulons[tf])} genes' )


    # get regulons from enriched motifs?
    #regulons = df2regulons(df)
    with ProgressBar():
        df = prune2df(dbs, modules, MOTIF_ANNOTATIONS_FNAME)
    # Create regulons from this table of enriched motifs.
    regulons = df2regulons(df)
    
    # STEP 4: Cellular enrichment (aka AUCell) from CLI
    #!pyscenic aucell \
    #{f_loom_path_scenic} \
    #reg.csv \
    #--output {f_pyscenic_output} \
    #--num_workers 20  
    

    # collect SCENIC AUCell output
    #lf = lp.connect( f_pyscenic_output, mode='r+', validate=False )
    #auc_mtx = pd.DataFrame( lf.ca.RegulonsAUC, index=lf.ca.CellID)
    #lf.close()
    
    auc_mtx = aucell(
        ex_matrix,
        signatures,
        auc_threshold=0.5,
        noweights=False,
        normalize=False,
        seed=None,
        num_workers=24)
    print(type(auc_mtx))
    print(auc_mtx.head())

    


    # END
    end = time.time()
    print(f'script finished in {(end-begin)//60} min {(end-begin)%60} sec')


