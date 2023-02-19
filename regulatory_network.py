#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
@file: regulatory_network.py
@time: 2023/Jan/08
@description: inference gene regulatory networks
@author: Yao LI
@email: liyao1@genomics.cn
@last modified by: Yao LI

change log:
    2023/01/08 init
'''

# python core modules
import os
import time
import csv
import warnings
import sys
from typing import Union

# third party modules
import glob
import anndata
import pandas as pd
import numpy as np
import seaborn as sns
import scanpy as sc
import matplotlib.pyplot as plt
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
from log_manager import LogManager
from algorithm_base import AlgorithmBase
from stereo.io.reader import read_gef
from plot_base import PlotBase
from stereo.core.stereo_exp_data import StereoExpData


class RegulatoryNetwork(AlgorithmBase):
    '''
    A gene regulatory network
    '''

    logger = LogManager(log_path='project.log', level='debug').get_logger(name='Stereo')

    def __init__(self):
        self.data = None
        self._genes = None  # list
        self._mtx = None  # pd.DataFrame
        self._regulons = None  # list, check
        self.modules = None # check
        self._auc_mtx = None  # check
        self.zscore_auc_mtx = None
        self.adjacencies = None  # pd.DataFrame
        self.tf_names = None # list

    @property
    def mtx(self):
        return self._mtx

    @mtx.setter
    def mtx(self, matrix):
        self._mtx = matrix

    @property
    def genes(self):
        return self._genes

    @genes.setter
    def genes(self, value):
        self._genes = value

    @property
    def regulons(self):
        return self._regulons

    @property
    def auc_mtx(self):
        return self._auc_mtx

    @classmethod
    def is_valid_exp_matrix(cls, mtx: pd.DataFrame):
        """check if the exp matrix is vaild for the grn pipeline"""
        return (all(isinstance(idx, str) for idx in mtx.index)
                and all(isinstance(idx, str) for idx in mtx.columns)
                and (mtx.index.nlevels == 1)
                and (mtx.columns.nlevels == 1))

    @classmethod
    def load_data(self, fn: str, bin_type='cell_bins'):
        """
        Loading input files, supported file formats:
            gef, gem, loom, h5ad, csv
        Recommended formats: h5ad, gef
        """
        self.logger.info('Loading expression data...')
        extension = os.path.splitext(fn)[1]
        if extension == '.csv':
            self.mtx = pd.read_csv(fn)
            self.genes = list(self.mtx.columns)
            return self.mtx, self.genes
        elif extension == '.loom':
            self.data = sc.read_loom(fn)
            self.genes = list(self.data.var_names)
            self.mtx = self.data.X
            return self.mtx, self.genes
        elif extension == 'h5ad':
            self.data = sc.read_h5ad(fn)
            self.genes = list(self.data.var_names)
            self.mtx = self.data.X
            return self.mtx, self.genes
        elif extension == '.gef':
            self.data = read_gef(file_path=fn, bin_type=bin_type)
            self.genes = self.data.gene_names
            self.mtx = self.data.to_df()
            return self.mtx, self.genes

    @staticmethod
    def _set_client(num_workers: int) -> Client:
        """

        :param num_workers:
        :return:
        """
        local_cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)
        custom_client = Client(local_cluster)
        return custom_client

    def grn_inference(self,
                      num_workers: int = 6,
                      verbose: bool = True,
                      fn: str = 'adj.csv') -> pd.DataFrame:
        """
        Inference of co-expression modules
        """
        self.logger.info('GRN inferencing...')
        begin1 = time.time()
        custom_client = RegulatoryNetwork._set_client(num_workers)
        self.adjacencies = grnboost2(self.mtx,
                                     tf_names=self.tf_names,
                                     gene_names=self.genes,
                                     verbose=verbose,
                                     client_or_address=custom_client)
        end1 = time.time()
        self.logger.info(f'GRN inference DONE in {(end1 - begin1) // 60} min {(end1 - begin1) % 60} sec')
        self.adjacencies.to_csv(fn, index=False)  # adj.csv, don't have to save into a file
        return self.adjacencies

    @staticmethod
    def _name(fname: str) -> str:
        return os.path.splitext(os.path.basename(fname))[0]

    @classmethod
    def load_database(cls, DATABASES_GLOB: str) -> list:
        cls.logger.info('Loading ranked databases...')
        db_fnames = glob.glob(DATABASES_GLOB)
        dbs = [RankingDatabase(fname=fname, name=RegulatoryNetwork._name(fname)) for fname in db_fnames]
        return dbs

    @classmethod
    def load_tfs(cls, fn):
        return load_tf_names(fn)

    def ctx_get_regulons(self,
                         rho_mask_dropouts: bool = False):
        """
        Inference of co-expression modules
        """
        begin2 = time.time()
        self.modules = list(
            modules_from_adjacencies(
                self.adjacencies,
                self.mtx,
                rho_mask_dropouts=rho_mask_dropouts
            )
        )
        end2 = time.time()
        self.logger.info(f'Regulon Prediction DONE in {(end2 - begin2) // 60} min {(end2 - begin2) % 60} sec')
        self.logger.info(f'generated {len(self.modules)} modules')
        return self.modules

    def prune(self,
              dbs: list,
              MOTIF_ANNOTATIONS_FNAME,
              num_workers: int = 6,
              is_prune: bool = True,
              rgn: str = 'regulons.csv'):
        """

        """
        if is_prune:
            with ProgressBar():
                df = prune2df(dbs, self.modules, MOTIF_ANNOTATIONS_FNAME, num_workers=num_workers)
            regulons = df2regulons(df)
            df.to_csv(rgn)  # motifs filename

            # alternative way of getting regulons, without creating df first
            regulons = self.prune(dbs, self.modules, MOTIF_ANNOTATIONS_FNAME)
            return regulons
        else:  # TODO: warning, is_prune setted as False
            pass

    def regulons_to_csv(self, fn: str = 'regulons.csv'):
        """
        Save regulons (df2regulons output) into a csv file.
        """
        rdict = {}
        for reg in self.regulons:
            targets = [target for target in reg.gene2weight]
            rdict[reg.name] = targets
        # Optional: join list of target genes
        for key in rdict.keys(): rdict[key] = ";".join(rdict[key])
        # Write to csv file
        with open(fn, 'w') as f:
            w = csv.writer(f)
            w.writerow(["Regulons", "Target_genes"])
            w.writerows(rdict.items())

    def auc_activity_level(self,
                           auc_thld,
                           num_workers,
                           fn='auc.csv') -> pd.DataFrame:
        begin4 = time.time()
        auc_mtx = aucell(self.mtx, self.regulons, auc_threshold=auc_thld, num_workers=num_workers)
        end4 = time.time()
        self.logger.info(f'Cellular Enrichment DONE in {(end4 - begin4) // 60} min {(end4 - begin4) % 60} sec')
        auc_mtx.to_csv(fn)
        return auc_mtx

    def save_to_loom(self, LOOM_FILE: str = 'output.loom'):
        export2loom(ex_mtx=self.mtx, auc_mtx=self.auc_mtx,
                    regulons=[r.rename(r.name.replace('(+)', ' (' + str(len(r)) + 'g)')) for r in self.regulons],
                    out_fname=LOOM_FILE)

    def uniq_genes(self):
        unique_adj_genes = set(self.adjacencies["TF"]).union(set(self.adjacencies["target"])) - set(self.mtx.columns)
        self.logger.info(f'find {len(unique_adj_genes) / len(set(self.mtx.columns))} unique genes')
        return unique_adj_genes

    def main(self):
        RESOURCES_FOLDER = "/dellfsqd2/ST_OCEAN/USER/liyao1/stereopy/resource"
        DATABASE_FOLDER = "/dellfsqd2/ST_OCEAN/USER/liyao1/stereopy/database/"
        DATABASES_GLOB = os.path.join(DATABASE_FOLDER,
                                      'mm10_10kbp_up_10kbp_down_full_tx_v10_clust.genes_vs_motifs.rankings.feather')
        MOTIF_ANNOTATIONS_FNAME = os.path.join(RESOURCES_FOLDER, 'motifs/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl')
        MM_TFS_FNAME = os.path.join(RESOURCES_FOLDER, 'tfs/test_mm_mgi_tfs.txt')
        SC_EXP_FNAME = os.path.join(RESOURCES_FOLDER, 'StereopyData/SS200000135TL_D1.cellbin.gef')

        grn = RegulatoryNetwork()
        # 0. Load StereoExpData file
        grn.load_data(SC_EXP_FNAME)
        # 1. load TF list
        tfs = grn.load_tfs(MM_TFS_FNAME)
        # 2. load the ranking databases
        dbs = grn.load_database(DATABASES_GLOB)
        # 3. GRN inference
        grn.grn_inference(num_workers=24)
        # 4. Regulon prediction aka cisTarget from CLI
        grn.ctx_get_regulons()
        grn.prune(dbs, MOTIF_ANNOTATIONS_FNAME, num_workers=24,)
        # 5: Cellular enrichment (aka AUCell) from CLI
        grn.auc_activity_level(auc_thld=0.5, num_workers=24)
        return grn


class PlotRegulatoryNetwork(PlotBase):
    @staticmethod
    def dotplot(data: Union[anndata.AnnData, StereoExpData]):
        """
        create a dotplot for the StereoExpData.
        a dotplot contains percent (of cells that) expressed (the genes) and average exression (of genes).
        :param data: a StereoExpData object or an Anndata object.
        :param: output: the output path. StereoExpData's output will be reset if the output is not None.

        :return:
        """
        if isinstance(data, anndata.AnnData):
            return sc.pl.dotplot(data)
        elif isinstance(data, StereoExpData):
            pass

    @staticmethod
    def auc_heatmap(auc_mtx, fn='auc_heatmap.png'):
        """

        :param auc_mtx:
        :param fn:
        :return:
        """
        plt.figsize = (8, 8)
        sns.clustermap(auc_mtx)
        plt.tightlayout()
        plt.savefig(fn)
