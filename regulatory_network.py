#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@file: regulatory_network.py
@time: 2023/Jan/08
@description: implement gene regulatory networks
@author: Yao LI
@email: liyao1@genomics.cn
@last modified by: Yao LI

change log:
    2023/01/08 init
'''

# python core modules
import os
import time
import sys

# third party modules
import glob
import json
import pandas as pd
import numpy as np
import seaborn as sns
import scanpy as sc
import loompy as lp
import matplotlib.pyplot as plt
from scanpy.pl import dotplot
from pyscenic.export import export2loom
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
from arboreto.utils import load_tf_names
from arboreto.algo import grnboost2
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.prune import prune2df, df2regulons
from pyscenic.utils import modules_from_adjacencies
from pyscenic.aucell import aucell

# modules in self project
from ..log_manager import logger
from .algorithm_base import AlgorithmBase
from stereo.io.reader import read_gef
from ..plots.plot_base import PlotBase
from stereo.core.stereo_exp_data import StereoExpData


class RegulatoryNetwork(AlgorithmBase):
    '''
    A network object
    '''
    logger = LogManager(log_path='project.log',level='debug').get_logger(name='Stereo')


    def __init__(self, matrix):
        self._expr_matrix = matrix
        self._pairs = [] #TF-gene pairs
        self._motif = [] #motif enrichment dataframe
        self._auc_matrix # TF activity level matrix


    @property
    def expr_matrix(self):
        return self._expr_matrix

    @expr_matrix.setter
    def expr_matrix(self, matrix):
        self._expr_matrix = matrix
    
    @staticmethod
    def is_valid_exp_matrix(mtx:pd.DataFrame):
        '''check if the exp matrix is vaild for the grn pipline'''
        return (all(isinstance(idx, str) for idx in mtx.index)
                and all(isinstance(idx, str) for idx in mtx.columns)
                and (mtx.index.nlevels == 1)
                and (mtx.columns.nlevels == 1))

    @staticmethod
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
        """
        Inference of co-expression modules
        """

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
        adjacencies.to_csv(fn, index=False) # adj.csv, don't have to save into a file
        return adjacencies    


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
                        rho_mask_dropouts:bool=False):
        """
        Inference of co-expression modules
        """
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
        return modules    
   

    def prune(modules, MOTIF_ANNOTATIONS_FNAME, num_workers=num_workers, is_prune:bool=True, rgn:str='regulons.csv'):
        if is_prune:
            with ProgressBar():
               df = prune2df(dbs, modules, MOTIF_ANNOTATIONS_FNAME, num_workers=num_workers) 
            regulons = df2regulons(df)
            df.to_csv(rgn)  # motifs filename

            # alternative way of getting regulons, without creating df first
            regulons = prune(dbs, modules, MOTIF_ANNOTATIONS_FNAME)
            return regulons
        else:  #TODO: warning, is_prune setted as False
            pass


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


    def auc_heatmap(auc_mtx, fn='auc_heatmap.png'):
        plt.figsize=(8,8)
        sns.clustermap(auc_mtx)
        plt.tightlayout()
        plt.savefig(fn)


    def save2loom(ex_matrix, auc_mtx, regulons, LOOM_FILE:str='output.loom'):
        export2loom(ex_mtx = ex_matrix, auc_mtx = auc_mtx, regulons = [r.rename(r.name.replace('(+)',' ('+str(len(r))+'g)')) for r in regulons], out_fname = LOOM_FILE)


    def uniq_genes(adjacencies, ex_matrix):
        unique_adj_genes = set(adjacencies["TF"]).union(set(adjacencies["target"])) - set(ex_matrix.columns)
        logger.info(f'find {len(unique_adj_genes)/len(set(ex_matrix.columns))} unique genes')
        return unique_adj_genes


    def main(self):
        '''
        A pipeline?
        '''
        pass


class PlotRegulatoryNetwork(PlotBase):
    @staticmethod
    def dotplot(data:StereoExpData):
        """
        create a dotplot for the StereoExpData.
        a dotplot contains percent (of cells that) expressed (the genes) and average exression (of genes).
        :param data: the StereoExpData object.
        :param: output: the output path. StereoExpData's output will be reset if the output is not None.

        :return:
        """
        pass


