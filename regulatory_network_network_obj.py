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
import csv
import warnings
from typing import Union

# third party modules
import glob
import anndata
import pandas as pd
import numpy as np
import seaborn as sns
import scanpy as sc
import matplotlib.pyplot as plt
from multiprocessing import cpu_count
from pyscenic.export import export2loom
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
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
    """
    A gene regulatory network
    """

    logger = LogManager(log_path='project.log', level='debug').get_logger(name='Stereo')

    def __init__(self):
        # input
        self.data = None
        self._genes = None  # list
        self._cells = None # list
        self._mtx = None  # pd.DataFrame

        self.tf_names = None  # list

        # network calculated attributes
        self._regulons = None  # list, check
        self.modules = None  # check
        self._auc_mtx = None  # check
        self.zscore_auc_mtx = None
        self.adjacencies = None  # pd.DataFrame

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
    def cells(self):
        return self._cells

    @cells.setter
    def cells(self, value):
        self._cells = value

    @property
    def regulons(self):
        return self._regulons

    @property
    def auc_mtx(self):
        return self._auc_mtx

    @staticmethod
    def is_valid_exp_matrix(mtx: pd.DataFrame):
        """
        check if the exp matrix is valid for the grn pipeline
        :param mtx:
        :return:
        """
        return (all(isinstance(idx, str) for idx in mtx.index)
                and all(isinstance(idx, str) for idx in mtx.columns)
                and (mtx.index.nlevels == 1)
                and (mtx.columns.nlevels == 1))

    def load_data(self, data: Union[StereoExpData, anndata.AnnData]):
        """

        :param data:
        :return:
        """
        self.data = data
        if isinstance(data, StereoExpData):
            self.genes = data.gene_names
            self.mtx = data.exp_matrix
            self.cells = data.cell_names
        elif isinstance(data, anndata.AnnData):
            self.mtx = data.X
            self.genes = list(data.var)

    def load_data_by_cluster(self, data: Union[StereoExpData, anndata.AnnData], cluster_name: str):
        """

        :param data:
        :param cluster_name:
        :return:
        """
        pass

    def read_file(self, fn: str, bin_type='cell_bins'):
        """
        Loading input files, supported file formats:
            gef, gem, loom, h5ad, csv
        Recommended formats: h5ad, gef
        :param fn:
        :param bin_type:
        :return:
        """
        self.logger.info('Loading expression data...')
        extension = os.path.splitext(fn)[1]
        if extension == '.csv':
            self.mtx = pd.read_csv(fn)
            self.genes = list(self.mtx.columns)
            self.logger.info(f'is valid expr matrix {RegulatoryNetwork.is_valid_exp_matrix(self.mtx)}')
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
                      num_workers: int,
                      verbose: bool = True,
                      fn: str = 'adj.csv') -> pd.DataFrame:
        """
        Inference of co-expression modules
        mtx:
           * a pandas DataFrame (rows=observations, columns=genes)
           * a dense 2D numpy.ndarray
           * a sparse scipy.sparse.csc_matrix
        :param num_workers:
        :param verbose:
        :param fn:
        :return:
        """
        if num_workers is None:
            num_workers = cpu_count()
        custom_client = RegulatoryNetwork._set_client(num_workers)
        self.adjacencies = grnboost2(self.mtx,
                                     tf_names=self.tf_names,
                                     gene_names=self.genes,
                                     verbose=verbose,
                                     client_or_address=custom_client)
        self.adjacencies.to_csv(fn, index=False)  # adj.csv, don't have to save into a file
        return self.adjacencies

    @staticmethod
    def _name(fname: str) -> str:
        """

        :param fname:
        :return:
        """
        return os.path.splitext(os.path.basename(fname))[0]

    @classmethod
    def load_database(cls, database_dir: str) -> list:
        """

        :param database_dir:
        :return:
        """
        cls.logger.info('Loading ranked databases...')
        db_fnames = glob.glob(database_dir)
        dbs = [RankingDatabase(fname=fname, name=RegulatoryNetwork._name(fname)) for fname in db_fnames]
        return dbs

    @classmethod
    def load_tfs(cls, fn: str)->list:
        """

        :param fn:
        :return:
        """
        with open(fn) as file:
            tfs_in_file = [line.strip() for line in file.readlines()]
        return tfs_in_file

    def ctx_get_regulons(self,
                         rho_mask_dropouts: bool = False):
        """
        Inference of co-expression modules
        :param rho_mask_dropouts:
        :return:
        """
        self.modules = list(
            modules_from_adjacencies(self.adjacencies, self.mtx, rho_mask_dropouts=rho_mask_dropouts)
        )
        return self.modules

    def prune(self,
              dbs: list,
              motif_anno_fn,
              num_workers: int,
              is_prune: bool = True,
              rgn: str = 'regulons.csv'):
        """

        :param dbs:
        :param motif_anno_fn:
        :param num_workers:
        :param is_prune:
        :param rgn:
        :return:
        """
        if num_workers is None:
            num_workers = cpu_count()
        if is_prune:
            with ProgressBar():
                df = prune2df(dbs, self.modules, motif_anno_fn, num_workers=num_workers)
            regulons = df2regulons(df)
            df.to_csv(rgn)  # motifs filename
            # alternative way of getting regulons, without creating df first
            regulons = self.prune(dbs, self.modules, motif_anno_fn)
            return regulons
        else:
            warnings.warn('if prune is set to False')

    def regulons_to_csv(self, fn: str = 'regulons.csv'):
        """
        Save regulons (df2regulons output) into a csv file.
        :param fn:
        :return:
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
                           auc_threshold: float,
                           num_workers: int,
                           save: bool = True,
                           fn='auc.csv') -> pd.DataFrame:
        """

        :param auc_threshold:
        :param num_workers:
        :param save:
        :param fn:
        :return:
        """
        if num_workers is None:
            num_workers = cpu_count()
        auc_mtx = aucell(self.mtx, self.regulons, auc_threshold=auc_threshold, num_workers=num_workers)
        if save:
            auc_mtx.to_csv(fn)
        return auc_mtx

    def save_to_loom(self, loom_fn: str = 'output.loom'):
        """

        :param loom_fn:
        :return:
        """
        export2loom(ex_mtx=self.mtx, auc_mtx=self.auc_mtx,
                    regulons=[r.rename(r.name.replace('(+)', ' (' + str(len(r)) + 'g)')) for r in self.regulons],
                    out_fname=loom_fn)

    def uniq_genes(self):
        """

        :return:
        """
        unique_adj_genes = set(self.adjacencies["TF"]).union(set(self.adjacencies["target"])) - set(self.mtx.columns)
        RegulatoryNetwork.logger.info(f'find {len(unique_adj_genes) / len(set(self.mtx.columns))} unique genes')
        return unique_adj_genes

    def main(self, data: Union[StereoExpData, anndata.AnnData], databases: str, motif_anno_fn: str, tfs_fn: str):
        """

        :param databases:
        :param motif_anno_fn:
        :param tfs_fn:
        :return:
        """
        #sc_exp_fname = os.path.join(resources_folder, 'StereopyData/SS200000135TL_D1.cellbin.gef')
        # 0. Load StereoExpData file
        #grn.read_file(sc_exp_fname)
        # 1. load TF list
        self.load_tfs(tfs_fn)
        # 2. load the ranking databases
        dbs = self.load_database(databases)
        # 3. GRN inference
        self.grn_inference(num_workers=24)
        # 4. Regulons prediction aka cisTarget
        self.ctx_get_regulons()
        self.prune(dbs, motif_anno_fn, num_workers=24)
        # 5: Cellular enrichment (aka AUCell)
        self.auc_activity_level(auc_threshold=0.5, num_workers=24)


class PlotRegulatoryNetwork(PlotBase):
    def _cal_percent_df(exp_matrix, cluster_meta, regulon, ct, cutoff=0):
        """
        Expression percent
        cell numbers
        """
        cells = cluster_meta['cluster' == ct]['cell']
        ct_exp = exp_matrix.iloc(cells)
        g_ct_exp = ct_exp[regulon]
        regulon_cell_num = g_ct_exp[g_ct_exp > cutoff].count()
        total_cell_num = 0
        return regulon_cell_num / total_cell_num

    def _cal_exp_df(exp_matrix, cluster_meta, regulon, ct):
        cells = cluster_meta['cluster' == ct]['cell']
        ct_exp = exp_matrix.iloc(cells)
        g_ct_exp = ct_exp[regulon]
        return np.mean(g_ct_exp)

    def dotplot2(StereoExpData, **kwargs):
        '''
        Intuitive way of visualizing how feature expression changes across different
        identity classes (clusters). The size of the dot encodes the percentage of
        cells within a class, while the color encodes the AverageExpression level
        across all cells within a class (blue is high).

        @param features Input vector of features, or named list of feature vectors
        if feature-grouped panels are desired
        '''
        pass

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

    @staticmethod
    def plot_reg_2d(zscore, coor, reg_name):
        # target_regs = list(auc_zscore.columns)
        # for reg in target_regs:
        #     if '(+)' not in reg:
        #         continue
        # prepare plotting data
        sub_zscore = zscore[['Cell', reg_name]]
        # sort data points by zscore (low to high)
        zorder = np.argsort(sub_zscore[reg_name].values)
        # plot cell/bin dot, x y coor
        sc = plt.scatter(coor['x'][zorder], coor['y'][zorder], c=sub_zscore[reg_name][zorder], marker='.',
                         edgecolors='none', cmap='plasma', lw=0)
        plt.box(False)
        plt.axis('off')
        plt.colorbar(sc, shrink=0.35)
        plt.savefig(f'{reg_name.split("(")[0]}_2d.png')
        plt.close()
