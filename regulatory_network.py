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


class GeneNetwork:
    def __init__(self):
        # input
        self.data = None
        self._genes = None  # list
        self._cells = None  # list
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


class InferenceRegulatoryNetwork(AlgorithmBase):
    """
    A gene regulatory network
    """

    logger = LogManager(log_path='project.log', level='debug').get_logger(name='Stereo')

    def __init__(self):
        # input
        self._data = None
        self._matrix = None  # pd.DataFrame
        # network calculated attributes
        self._regulons = None  # list, check
        self._auc_mtx = None  # check
        self._adjacencies = None  # pd.DataFrame

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, value):
        self._network = value

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self._matrix = value

    @property
    def adjacencies(self):
        return self._adjacencies

    @adjacencies.setter
    def genes(self, value):
        self._adjacencies = value

    @property
    def regulons(self):
        return self._regulons

    @regulons.setter
    def regulons(self, value):
        self._regulons = value

    @property
    def auc_mtx(self):
        return self._auc_mtx

    @auc_mtx.setter
    def auc_mtx(self, value):
        self._auc_mtx = value

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
        self.network.data = data
        if isinstance(data, StereoExpData):
            self.network.genes = data.gene_names
            self.network.mtx = data.exp_matrix
            self.network.cells = data.cell_names
        elif isinstance(data, anndata.AnnData):
            self.network.mtx = data.X
            self.network.genes = list(data.var)

    @classmethod
    def load_data_by_cluster(cls, data: Union[StereoExpData, anndata.AnnData], cluster_label: str, target_clusters:list):
        """

        :param data:
        :param cluster_label:
        :return:
        """
        if isinstance(data, anndata.AnnData):
            return data[data.obs[cluster_label].isin(target_clusters)]

    @classmethod
    def read_file(cls, fn: str, bin_type='cell_bins'):
        """
        Loading input files, supported file formats:
            * gef
            * gem
            * loom
            * h5ad
            * csv
        Recommended formats:
            * h5ad
            * gef
        :param fn:
        :param bin_type:
        :return:
        """
        #self.logger.info('Loading expression data...')
        extension = os.path.splitext(fn)[1]
        if extension == '.csv':
            cls.mtx = pd.read_csv(fn)
            cls.genes = list(cls.mtx.columns)
            #self.logger.info(f'is valid expr matrix {RegulatoryNetwork.is_valid_exp_matrix(self.mtx)}')
            return cls.mtx, cls.genes
        elif extension == '.loom':
            cls.data = sc.read_loom(fn)
            cls.genes = list(cls.data.var_names)
            cls.mtx = cls.data.X
            return cls.mtx, cls.genes
        elif extension == 'h5ad':
            cls.data = sc.read_h5ad(fn)
            cls.genes = list(cls.data.var_names)
            cls.mtx = cls.data.X
            return cls.mtx, cls.genes
        elif extension == '.gef':
            cls.data = read_gef(file_path=fn, bin_type=bin_type)
            cls.genes = cls.data.gene_names
            cls.mtx = cls.data.to_df()
            return cls.mtx, cls.genes

    @staticmethod
    def _set_client(num_workers: int) -> Client:
        """

        :param num_workers:
        :return:
        """
        local_cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1)
        custom_client = Client(local_cluster)
        return custom_client

    @classmethod
    def grn_inference(cls,
                      matrix,
                      tf_names: list,
                      genes,
                      num_workers: int,
                      verbose: bool = True,
                      fn: str = 'adj.csv') -> pd.DataFrame:
        """
        Inference of co-expression modules
        mtx:
           * pandas DataFrame (rows=observations, columns=genes)
           * dense 2D numpy.ndarray
           * sparse scipy.sparse.csc_matrix
        :param num_workers:
        :param verbose:
        :param fn:
        :return:
        """
        #global matrix
        if num_workers is None:
            num_workers = cpu_count()
        custom_client = InferenceRegulatoryNetwork._set_client(num_workers)
        # if isinstance(data, StereoExpData):
        #     matrix = data.exp_matrix
        # elif isinstance(data, anndata.AnnData):
        #     matrix = data.X
        # assert matrix is not None
        adjacencies = grnboost2(matrix,
                                tf_names=tf_names,
                                gene_names=genes,
                                verbose=verbose,
                                client_or_address=custom_client)
        adjacencies.to_csv(fn, index=False)  # adj.csv, don't have to save into a file
        return adjacencies

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
        #cls.logger.info('Loading ranked databases...')
        db_fnames = glob.glob(database_dir)
        dbs = [RankingDatabase(fname=fname, name=InferenceRegulatoryNetwork._name(fname)) for fname in db_fnames]
        return dbs

    @classmethod
    def load_tfs(cls, fn: str) -> list:
        """

        :param fn:
        :return:
        """
        with open(fn) as file:
            tfs_in_file = [line.strip() for line in file.readlines()]
        return tfs_in_file

    @classmethod
    def ctx_get_regulons(cls,
                         adjacencies,
                         matrix,
                         rho_mask_dropouts: bool = False):
        """
        Inference of co-expression modules
        :param rho_mask_dropouts:
        :return:
        """
        modules = list(
            modules_from_adjacencies(adjacencies, matrix, rho_mask_dropouts=rho_mask_dropouts)
        )
        return modules

    @classmethod
    def prune(cls,
              modules,
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
                df = prune2df(dbs, modules, motif_anno_fn, num_workers=num_workers)
            regulons = df2regulons(df)
            df.to_csv(rgn)  # motifs filename
            # alternative way of getting regulons, without creating df first
            regulons = cls.prune(dbs, modules, motif_anno_fn)
            return regulons
        else:
            warnings.warn('if prune is set to False')

    @classmethod
    def regulons_to_csv(cls, fn: str = 'regulons.csv'):
        """
        Save regulons (df2regulons output) into a csv file.
        :param fn:
        :return:
        """
        rdict = {}
        for reg in cls.regulons:
            targets = [target for target in reg.gene2weight]
            rdict[reg.name] = targets
        # Optional: join list of target genes
        for key in rdict.keys(): rdict[key] = ";".join(rdict[key])
        # Write to csv file
        with open(fn, 'w') as f:
            w = csv.writer(f)
            w.writerow(["Regulons", "Target_genes"])
            w.writerows(rdict.items())

    @classmethod
    def auc_activity_level(cls,
                           matrix,
                           regulons,
                           auc_threshold: float,
                           num_workers: int,
                           save: bool = True,
                           fn='auc.csv') -> pd.DataFrame:
        """

        :param matrix:
        :param regulons:
        :param auc_threshold:
        :param num_workers:
        :param save:
        :param fn:
        :return:
        """
        if num_workers is None:
            num_workers = cpu_count()
        auc_mtx = aucell(matrix, regulons, auc_threshold=auc_threshold, num_workers=num_workers)
        if save:
            auc_mtx.to_csv(fn)
        return auc_mtx

    @classmethod
    def save_to_loom(cls, matrix, auc_matrix, regulons, loom_fn: str = 'output.loom'):
        """

        :param matrix:
        :param auc_matrix:
        :param regulons:
        :param loom_fn:
        :return:
        """
        export2loom(ex_mtx=matrix, auc_mtx=auc_matrix,
                    regulons=[r.rename(r.name.replace('(+)', ' (' + str(len(r)) + 'g)')) for r in regulons],
                    out_fname=loom_fn)

    @classmethod
    def uniq_genes(cls, adjacencies, matrix:pd.DataFrame):
        """

        :param adjacencies:
        :param matrix:
        :return:
        """
        unique_adj_genes = set(adjacencies["TF"]).union(set(adjacencies["target"])) - set(matrix.columns)
        #RegulatoryNetwork.logger.info(f'find {len(unique_adj_genes) / len(set(cls.mtx.columns))} unique genes')
        return unique_adj_genes

    @classmethod
    def main(self,
             data: Union[StereoExpData, anndata.AnnData],
             databases: str,
             motif_anno_fn: str,
             tfs_fn: str,
             num_workers: int):
        """"""
        global matrix
        if isinstance(data, StereoExpData):
            matrix = data.exp_matrix
        elif isinstance(data, anndata.AnnData):
            matrix = data.X
        assert matrix is not None

        # 1. load TF list
        tfs = self.load_tfs(tfs_fn)
        # 2. load the ranking databases
        dbs = self.load_database(databases)
        # 3. GRN inference
        adjacencies = self.grn_inference(matrix, genes=data.var_names, tf_names=tfs, num_workers=num_workers)
        # 4. Regulons prediction aka cisTarget
        modules = self.ctx_get_regulons(adjacencies, matrix)
        regulons = self.prune(modules, dbs, motif_anno_fn, num_workers=24)
        # 5: Cellular enrichment (aka AUCell)
        auc_mtx = self.auc_activity_level(matrix, regulons, auc_threshold=0.5, num_workers=num_workers)
        return adjacencies, regulons, auc_mtx


class PlotRegulatoryNetwork(PlotBase):
    @staticmethod
    def _cal_percent_df(exp_matrix, cluster_meta, regulon, ct, cutoff=0):
        """
        Expression percent
        cell numbers
        :param exp_matrix:
        :param cluster_meta:
        :param regulon:
        :param ct:
        :param cutoff:
        :return:
        """
        cells = cluster_meta['cluster' == ct]['cell']
        ct_exp = exp_matrix.iloc(cells)
        g_ct_exp = ct_exp[regulon]
        regulon_cell_num = g_ct_exp[g_ct_exp > cutoff].count()
        total_cell_num = 0
        return regulon_cell_num / total_cell_num

    @staticmethod
    def _cal_exp_df(exp_matrix, cluster_meta, regulon, ct):
        """

        :param exp_matrix:
        :param cluster_meta:
        :param regulon:
        :param ct:
        :return:
        """
        cells = cluster_meta['cluster' == ct]['cell']
        ct_exp = exp_matrix.iloc(cells)
        g_ct_exp = ct_exp[regulon]
        return np.mean(g_ct_exp)

    @staticmethod
    def dotplot_stereo(StereoExpData, **kwargs):
        """
        Intuitive way of visualizing how feature expression changes across different
        identity classes (clusters). The size of the dot encodes the percentage of
        cells within a class, while the color encodes the AverageExpression level
        across all cells within a class (blue is high).

        :param StereoExpData:
        :param kwargs: features Input vector of features, or named list of feature vectors
        if feature-grouped panels are desired
        :return:
        """
        pass

    @staticmethod
    def dotplot_anndata(data: Union[anndata.AnnData, StereoExpData],
                        gene_names,
                        cluster_label: str,
                        save: bool = True):
        """
        create a dotplot for Anndata object.
        a dotplot contains percent (of cells that) expressed (the genes) and average exression (of genes).

        :param data: gene data
        :param gene_names: interested gene names
        :param cluster_label: label of clustering output
        :param save: if save plot into a file
        :return: plt axe object

        e.g.
            there is an anndata
        """
        if isinstance(data, anndata.AnnData):
            return sc.pl.dotplot(data, var_names = gene_names, groupby=cluster_label, save=save)
        elif isinstance(data, StereoExpData):
            print('for StereoExpData object, please use function: dotplot_stereo')

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
    def plot_reg_2d(auc_zscore, cell_coor, reg_name):
        """
        
        :param auc_zscore:
        :param cell_coor:
        :param reg_name:
        :return:
        """
        # prepare plotting data
        sub_zscore = auc_zscore[['Cell', reg_name]]
        # sort data points by zscore (low to high)
        zorder = np.argsort(sub_zscore[reg_name].values)
        # plot cell/bin dot, x y coor
        sc = plt.scatter(cell_coor['x'][zorder], cell_coor['y'][zorder], c=sub_zscore[reg_name][zorder], marker='.',
                         edgecolors='none', cmap='plasma', lw=0)
        plt.box(False)
        plt.axis('off')
        plt.colorbar(sc, shrink=0.35)
        plt.savefig(f'{reg_name.split("(")[0]}.png')
        plt.close()

    @staticmethod
    def multi_reg_2d(auc_zscore, cell_coor, target_regs):
        """

        :param auc_zscore:
        :param cell_coor:
        :param target_regs:
        :return:
        """
        for reg in target_regs:
            if PlotRegulatoryNetwork.is_regulon(reg):
                PlotRegulatoryNetwork.plot_reg_2d(auc_zscore, cell_coor, reg)

    @staticmethod
    def is_regulon(reg):
        """

        :param reg:
        :return:
        """
        if '(+)' in reg or '(-)' in reg:
            return True



