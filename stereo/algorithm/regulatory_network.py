#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@file: regulatory_network.py
@time: 2023/Jan/08
@description: inference gene regulatory networks
@author: Yao LI
@email: liyao1@genomics.cn
@last modified by: Yao LI

change log:
    2023/01/08 init
"""

# python core modules
import os
import sys
import csv
import warnings
from typing import Union

# third party modules
import json
import glob
import anndata
import hotspot
import scipy.sparse
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
from pyscenic.cli.utils import load_signatures
from pyscenic.rss import regulon_specificity_scores
from pyscenic.aucell import aucell

# modules in self project
from log_manager import logger
from algorithm_base import AlgorithmBase
from stereo.io.reader import read_gef
from plot_base import PlotBase
from stereo.core.stereo_exp_data import StereoExpData


def _name(fname: str) -> str:
    """
    Extract file name (without path and extension)
    :param fname:
    :return:
    """
    return os.path.splitext(os.path.basename(fname))[0]


class InferenceRegulatoryNetwork(AlgorithmBase):
    """
    Algorithms to inference Gene Regulatory Networks (GRN)
    """

    def __init__(self, data):
        super(InferenceRegulatoryNetwork, self).__init__(data)
        # input
        self._data = data
        self._matrix = None  # pd.DataFrame
        self._gene_names = []
        self._cell_names = []

        self.load_data_info()

        self._tfs = []

        # network calculated attributes
        self._regulon_list = None  # list
        self._auc_mtx = None
        self._adjacencies = None  # pd.DataFrame
        self._regulon_dict = None

        # other settings
        # self._params = {'hotspot': {
        #                     'num_workers': None,
        #                     'rank_threshold': 1500,
        #                     'prune_auc_threshold': 0.001,
        #                     'nes_threshold': 3.0,
        #                     'motif_similarity_fdr': 0.1,
        #                     'auc_threshold: 0.5
        #                 },
        #                 'grnboost': {
        #                     'num_workers': None,
        #                     'rank_threshold': 1500,
        #                     'prune_auc_threshold': 0.5,
        #                     'nes_threshold': 3.0,
        #                     'motif_similarity_fdr': 0.05,
        #                     'auc_threshold': 0.5
        #                 },
        #                 'scoexp': {
        #                     'num_workers': None,
        #                     'rank_threshold': 1500,
        #                     'prune_auc_threshold': 0.001,
        #                     'nes_threshold': 3.0,
        #                     'motif_similarity_fdr': 0.05,
        #                     'auc_threshold': 0.5
        #                 }}

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data: Union[StereoExpData, anndata.AnnData]):
        self._data = data
        self.load_data_info()

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self._matrix = value

    @property
    def gene_names(self):
        return self._gene_names

    @gene_names.setter
    def gene_names(self, value):
        self._gene_names = value

    @property
    def cell_names(self):
        return self._cell_names

    @cell_names.setter
    def cell_names(self, value):
        self._cell_names = value

    @property
    def adjacencies(self):
        return self._adjacencies

    @adjacencies.setter
    def adjacencies(self, value):
        self._adjacencies = value

    @property
    def regulon_list(self):
        return self._regulon_list

    @regulon_list.setter
    def regulon_list(self, value):
        self._regulon_list = value

    @property
    def regulon_dict(self):
        return self._regulon_dict

    @regulon_dict.setter
    def regulon_dict(self, value):
        self._regulon_dict = value

    @property
    def auc_mtx(self):
        return self._auc_mtx

    @auc_mtx.setter
    def auc_mtx(self, value):
        self._auc_mtx = value

    def load_data_info(self):
        """"""
        if isinstance(self._data, StereoExpData):
            self._matrix = self._data.exp_matrix
            self._gene_names = self._data.gene_names
            self._cell_names = self._data.cell_names
        elif isinstance(self._data, anndata.AnnData):
            self._matrix = self._data.X
            self._gene_names = self._data.var_names
            self._cell_names = self._data.obs_names

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

    # Data loading methods
    @staticmethod
    def read_file(fn: str, bin_type='cell_bins'):
        """
        Loading input files, supported file formats:
            * gef
            * gem
            * loom
            * h5ad
        Recommended formats:
            * h5ad
            * gef
        :param fn:
        :param bin_type:
        :return:

        Example:
            grn.read_file('test.gef', bin_type='bins')
            or
            grn.read_file('test.h5ad')
        """
        logger.info('Loading expression data...')
        extension = os.path.splitext(fn)[1]
        logger.info(f'file extension is {extension}')
        if extension == '.csv':
            logger.error('read_file method does not support csv files')
            raise TypeError('this method does not support csv files, '
                            'please read this file using functions outside of the InferenceRegulatoryNetwork class, '
                            'e.g. pandas.read_csv')
        elif extension == '.loom':
            data = sc.read_loom(fn)
            return data
        elif extension == '.h5ad':
            data = sc.read_h5ad(fn)
            return data
        elif extension == '.gef':
            data = read_gef(file_path=fn, bin_type=bin_type)
            return data

    @staticmethod
    def load_anndata_by_cluster(fn: str,
                                cluster_label: str,
                                target_clusters: list) -> anndata.AnnData:
        """
        When loading anndata, only load in wanted clusters
        One must perform Clustering beforehand
        :param fn: data file name
        :param cluster_label: where the clustering results are stored
        :param target_clusters: a list of interested cluster names
        :return:

        Example:
            sub_data = load_anndata_by_cluster(data, 'psuedo_class', ['HBGLU9'])
        """
        data = InferenceRegulatoryNetwork.read_file(fn)
        if isinstance(data, anndata.AnnData):
            return data[data.obs[cluster_label].isin(target_clusters)]
        else:
            raise TypeError('data must be anndata.Anndata object')

    @staticmethod
    def load_stdata_by_cluster(data: StereoExpData,
                               meta: pd.DataFrame,
                               cluster_label: str,
                               target_clusters: list) -> scipy.sparse.csc_matrix:
        """

        :param data:
        :param meta:
        :param cluster_label:
        :param target_clusters:
        :return:
        """
        return data.exp_matrix[meta[cluster_label].isin(target_clusters)]

    @staticmethod
    def read_motif_file(fname):
        """

        :param fname:
        :return:
        """
        df = pd.read_csv(fname, sep=',', index_col=[0, 1], header=[0, 1], skipinitialspace=True)
        df[('Enrichment', 'Context')] = df[('Enrichment', 'Context')].apply(lambda s: eval(s))
        df[('Enrichment', 'TargetGenes')] = df[('Enrichment', 'TargetGenes')].apply(lambda s: eval(s))
        return df

    @staticmethod
    def load_tfs(fn: str) -> list:
        """

        :param fn:
        :return:
        """
        with open(fn) as file:
            tfs_in_file = [line.strip() for line in file.readlines()]
        return tfs_in_file

    # -------------------------------------------#
    #             NETWORK INFERENCE              #
    # -------------------------------------------#
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
                      matrix,
                      tf_names,
                      genes: list,
                      num_workers: int,
                      verbose: bool = True,
                      cache: bool = True,
                      save: bool = True,
                      fn: str = 'adj.csv',
                      **kwargs) -> pd.DataFrame:
        """
        Inference of co-expression modules
        :param matrix:
            * pandas DataFrame (rows=observations, columns=genes)
            * dense 2D numpy.ndarray
            * sparse scipy.sparse.csc_matrix
        :param tf_names: list of target TFs or all
        :param genes: list of interested genes
        :param num_workers: number of thread
        :param verbose: if print out running details
        :param cache:
        :param save: if save adjacencies result into a file
        :param fn: adjacencies file name
        :return:

        Example:

        """
        if cache and os.path.isfile(fn):
            logger.info(f'cached file {fn} found')
            adjacencies = pd.read_csv(fn)
            self.adjacencies = adjacencies
            return adjacencies
        else:
            logger.info('cached file not found, running grnboost2 now')

        if num_workers is None:
            num_workers = cpu_count()
        custom_client = InferenceRegulatoryNetwork._set_client(num_workers)
        adjacencies = grnboost2(matrix,
                                tf_names=tf_names,
                                gene_names=genes,
                                verbose=verbose,
                                client_or_address=custom_client,
                                **kwargs)
        logger.info('Network Inference DONE')

        if save:
            adjacencies.to_csv(fn, index=False)
        self.adjacencies = adjacencies
        return adjacencies

    def uniq_genes(self, adjacencies):
        """
        Detect unique genes
        :param adjacencies:
        :return:
        """
        df = self._data.to_df()
        unique_adj_genes = set(adjacencies["TF"]).union(set(adjacencies["target"])) - set(df.columns)
        logger.info(f'find {len(unique_adj_genes) / len(set(df.columns))} unique genes')
        return unique_adj_genes

    @staticmethod
    def input_hotspot(data: StereoExpData):
        """
        Extract needed information to construct a Hotspot instance from StereoExpData data
        :param data:
        :return: a dictionary
        """
        # 3. use dataframe and position array, StereoExpData as well
        counts = data.to_df().T  # gene x cell
        position = data.position
        num_umi = counts.sum(axis=0)  # total counts for each cell
        # Filter genes
        gene_counts = (counts > 0).sum(axis=1)
        valid_genes = gene_counts >= 50
        counts = counts.loc[valid_genes]
        return {'counts': counts, 'num_umi': num_umi, 'position': position}

    @staticmethod
    def hotspot_matrix(count_data,
                       model='bernoulli',
                       position,
                       num_umi,
                       tree=None,
                       weighted_graph: bool = False,
                       n_neighbors=30,
                       fdr_threshold: float = 0.05,
                       tf_list: list = None,
                       save=True,
                       jobs=None,
                       fn: str = 'adj.csv') -> pd.DataFrame:
        """
        Inference of co-expression modules via hotspot method
        :param count_data: Count matrix (shape is cells by genes)
        :param model: Specifies the null model to use for gene expression.
            Valid choices are:
                * 'danb': Depth-Adjusted Negative Binomial
                * 'bernoulli': Models probability of detection
                * 'normal': Depth-Adjusted Normal
                * 'none': Assumes data has been pre-standardized
        :param position: cell coordinates, np.array or pd.DataFrame
        :param num_umi: total counts for each cell
        :param tree: Root tree node.  Can be created using ete3.Tree
        :param weighted_graph: Whether or not to create a weighted graph
        :param n_neighbors: Neighborhood size
        :param fdr_threshold: Correlation threshold at which to stop assigning genes to modules
        :param tf_list: predefined TF names
        :param save: if save results onto disk
        :param jobs: Number of parallel jobs to run
        :param fn: output file name
        :return: A dataframe, local correlation Z-scores between genes (shape is genes x genes)
        """
        # hotspot_data = InferenceRegulatoryNetwork.input_hotspot(data)
        # hs = hotspot.Hotspot.legacy_init(hotspot_data['counts'],
        #                                  model=model,
        #                                  latent=hotspot_data['position'],
        #                                  umi_counts=hotspot_data['num_umi'],
        #                                  tree=tree)
        hs = hotspot.Hotspot.legacy_init(count_data,
                                         model=model,
                                         latent=position,
                                         umi_counts=num_umi,
                                         tree=tree)
        hs.create_knn_graph(weighted_graph=weighted_graph, n_neighbors=n_neighbors)

        hs_results = hs.compute_autocorrelations(jobs=jobs)

        hs_genes = hs_results.loc[hs_results.FDR < fdr_threshold].index  # Select genes
        local_correlations = hs.compute_local_correlations(hs_genes, jobs=jobs)  # jobs for parallelization
        logger.info('Network Inference DONE')
        logger.info(f'Hotspot: create {local_correlations.shape[0]} features')
        local_correlations.to_csv('local_correlations.csv')

        # subset by TFs
        if tf_list:
            common_tf_list = list(set(tf_list).intersection(set(local_correlations.columns)))
            logger.info(f'detected {len(common_tf_list)} predefined TF in data')
            assert len(common_tf_list) > 0, 'predefined TFs not found in data'
        else:
            common_tf_list = local_correlations.columns

        # reshape matrix
        local_correlations['TF'] = local_correlations.columns
        local_correlations = local_correlations.melt(id_vars=['TF'])
        local_correlations.columns = ['TF', 'target', 'importance']
        local_correlations = local_correlations[local_correlations.TF.isin(common_tf_list)]
        # remove pairs if TF = target
        local_correlations = local_correlations[local_correlations.TF != local_correlations.target]

        if save:
            local_correlations.to_csv(fn, index=False)
        return local_correlations

    @staticmethod
    def load_database(database_dir: str) -> list:
        """
        Load ranked database
        :param database_dir:
        :return:
        """
        logger.info('Loading ranked databases...')
        db_fnames = glob.glob(database_dir)
        dbs = [RankingDatabase(fname=fname, name=_name(fname)) for fname in db_fnames]
        return dbs

    # -------------------------------------------#
    #            MODULE GENERATION               #
    # -------------------------------------------#
    @staticmethod
    def get_modules(adjacencies: pd.DataFrame,
                    matrix,
                    rho_mask_dropouts: bool = False,
                    **kwargs):
        """
        Inference of co-expression modules

        :param adjacencies:
        :param matrix:
            * pandas DataFrame (rows=observations, columns=genes)
            * dense 2D numpy.ndarray
            * sparse scipy.sparse.csc_matrix
        :param rho_mask_dropouts:
        :return:
        """
        modules = list(
            modules_from_adjacencies(adjacencies, matrix, rho_mask_dropouts=rho_mask_dropouts, **kwargs)
        )
        return modules

    def prune_modules(self,
                      modules: list,
                      dbs: list,
                      motif_anno_fn: str,
                      num_workers: int,
                      is_prune: bool = True,
                      cache: bool = True,
                      save: bool = True,
                      fn: str = 'motifs.csv',
                      **kwargs):
        """
        First, calculate a list of enriched motifs and the corresponding target genes for all modules.
        Then, create regulon_list from this table of enriched motifs.
        :param modules:
        :param dbs:
        :param motif_anno_fn:
        :param num_workers:
        :param is_prune:
        :param cache:
        :param save:
        :param fn:
        :return:
        """
        if cache and os.path.isfile(fn):
            logger.info(f'cached file {fn} found')
            df = self.read_motif_file(fn)
            regulon_list = df2regulons(df)
            # alternative:
            # regulon_list = load_signatures(fn)
            self.regulon_list = regulon_list
            return regulon_list
        else:
            logger.info('cached file not found, running prune modules now')

        if num_workers is None:
            num_workers = cpu_count()
        if is_prune:
            with ProgressBar():
                df = prune2df(dbs, modules, motif_anno_fn, num_workers=num_workers, **kwargs)
            regulon_list = df2regulons(df)
            self.regulon_list = regulon_list
            logger.info('Prune DONE')

            if save:
                df.to_csv(fn)  # motifs filename
                #self.regulons_to_json(regulon_list)

            return regulon_list
        else:
            logger.warning('if prune_modules is set to False')

    @staticmethod
    def get_regulon_dict(regulon_list: list) -> dict:
        """
        Form dictionary of { TF : Target } pairs from 'pyscenic ctx' output.
        :param regulon_list:
        :return:
        """
        regulon_dict = {}
        for reg in regulon_list:
            targets = [target for target in reg.gene2weight]
            regulon_dict[reg.name] = targets
        return regulon_dict

    # -------------------------------------------#
    #    MOTIF ENRICHMENT & REGULON PREDICTION   #
    # -------------------------------------------#
    def auc_activity_level(self,
                           matrix,
                           regulons: list,
                           auc_threshold: float,
                           num_workers: int,
                           cache: bool = True,
                           save: bool = True,
                           fn='auc.csv',
                           **kwargs) -> pd.DataFrame:
        """

        :param matrix:
            * pandas DataFrame (rows=observations, columns=genes)
            * dense 2D numpy.ndarray
            * sparse scipy.sparse.csc_matrix
        :param regulons: list of ctxcore.genesig.Regulon objects
        :param auc_threshold:
        :param num_workers:
        :param cache:
        :param save:
        :param fn:
        :return:
        """
        if cache and os.path.isfile(fn):
            logger.info(f'cached file {fn} found')
            auc_mtx = pd.read_csv(fn, index_col=0)  # important! cell must be index, not one of the column
            self.auc_mtx = auc_mtx
            return auc_mtx
        else:
            logger.info('cached file not found, calculating auc_activity_level now')

        if num_workers is None:
            num_workers = cpu_count()

        auc_mtx = aucell(matrix, regulons, auc_threshold=auc_threshold, num_workers=num_workers, **kwargs)
        self.auc_mtx = auc_mtx

        if save:
            auc_mtx.to_csv(fn)
        return auc_mtx

    # -------------------------------------------#
    #               OTHER METHODS                #
    # -------------------------------------------#
    def regulons_to_json(self, regulon_list: list, fn='regulons.json'):
        """
        Write regulon dictionary into json file
        :param regulon_list:
        :param fn:
        :return:
        """
        regulon_dict = self.get_regulon_dict(regulon_list)
        with open(fn, 'w') as f:
            json.dump(regulon_dict, f, indent=4)

    def regulons_to_csv(self, regulon_list: list, fn: str = 'regulon_list.csv'):
        """
        Save regulon_list (df2regulons output) into a csv file.
        :param regulon_list:
        :param fn:
        :return:
        """
        regulon_dict = self.get_regulon_dict(regulon_list)
        # Optional: join list of target genes
        for key in regulon_dict.keys(): regulon_dict[key] = ";".join(regulon_dict[key])
        # Write to csv file
        with open(fn, 'w') as f:
            w = csv.writer(f)
            w.writerow(["Regulons", "Target_genes"])
            w.writerows(regulon_dict.items())

    def to_loom(self, matrix: pd.DataFrame, auc_matrix: pd.DataFrame, regulons: list, loom_fn: str = 'output.loom'):
        """
        Save GRN results in one loom file
        :param matrix:
        :param auc_matrix:
        :param regulons:
        :param loom_fn:
        :return:
        """
        export2loom(ex_mtx=matrix, auc_mtx=auc_matrix,
                    regulons=[r.rename(r.name.replace('(+)', ' (' + str(len(r)) + 'g)')) for r in regulons],
                    out_fname=loom_fn)

    def to_cytoscape(self,
                     regulons: list,
                     adjacencies: pd.DataFrame,
                     tf: str,
                     fn: str = 'cytoscape.txt'):
        """
        Save GRN result of one TF, into Cytoscape format for down stream analysis
        :param regulons: list of regulon objects, output of prune step
        :param adjacencies: adjacencies matrix
        :param tf: one target TF name
        :param fn: output file name
        :return:

        Example:
            grn.to_cytoscape(regulons, adjacencies, 'Gnb4', 'Gnb4_cytoscape.txt')
        """
        # get TF data
        regulon_dict = self.get_regulon_dict(regulons)
        sub_adj = adjacencies[adjacencies.TF == tf]
        targets = regulon_dict[f'{tf}(+)']
        # all the target genes of the TF
        sub_df = sub_adj[sub_adj.target.isin(targets)]
        sub_df.to_csv(fn, index=False, sep='\t')

    # -------------------------------------------#
    #           GRN pipeline main logic          #
    # -------------------------------------------#
    def main(self,
             databases: str,
             motif_anno_fn: str,
             tfs_fn,
             method: str = 'grnboost',
             target_genes=None,
             cache=False,
             num_workers=None,
             save=True,
             prefix: str = 'project'):
        """
        :param method:
        :param databases:
        :param motif_anno_fn:
        :param tfs_fn:
        :param target_genes:
        :param cache:
        :param num_workers:
        :param save:
        :param prefix:
        :return:
        """
        global adjacencies
        matrix = self._matrix
        df = self._data.to_df()
        if num_workers is None:
            num_workers = cpu_count()

        if target_genes is None:
            target_genes = self._gene_names

        # 1. load TF list
        if tfs_fn is None:
            tfs = 'all'
            # tfs = self._gene_names
        else:
            tfs = self.load_tfs(tfs_fn)

        # 2. load the ranking databases
        dbs = self.load_database(databases)

        # 3. GRN inference
        if method == 'grnboost':
            adjacencies = self.grn_inference(matrix, genes=target_genes, tf_names=tfs, num_workers=num_workers,
                                             cache=False, save=save, fn=f'{prefix}_adj.csv')
        elif method == 'hotspot':
            adjacencies = self.hotspot_matrix(self.data, position=, num_umi=, tf_list=tfs, jobs=num_workers)
        # adj_col = adjacencies.columns
        modules = self.get_modules(adjacencies, df)

        # 4. Regulons prediction aka cisTarget
        regulons = self.prune_modules(modules, dbs, motif_anno_fn, num_workers=num_workers, save=save, cache=cache,
                                      fn=f'{prefix}_motifs.csv', rank_threshold=1500, auc_threshold=0.001,
                                      nes_threshold=3.0, motif_similarity_fdr=0.1)
        self.regulon_dict = self.get_regulon_dict(regulons)

        # 5: Cellular enrichment (aka AUCell)
        auc_matrix = self.auc_activity_level(df, regulons, auc_threshold=0.5, num_workers=num_workers, cache=cache,
                                             save=save, fn=f'{prefix}_auc.csv')

        # save results
        if save:
            self.regulons_to_json(regulons, fn=f'{prefix}_regulons.json')
            self.to_loom(df, auc_matrix, regulons)
            # self.to_cytoscape(regulons, adjacencies, 'Zfp354c')


class PlotRegulatoryNetwork(PlotBase):
    """
    Plot Gene Regulatory Networks related plots
    """

    def __init__(self, data):
        super(PlotRegulatoryNetwork, self).__init__(data)
        self._regulon_list = None
        self._auc_mtx = None
        self._regulon_dict = None

    @property
    def regulon_list(self):
        return self._regulon_list

    @regulon_list.setter
    def regulon_list(self, value):
        self._regulon_list = value

    @property
    def regulon_dict(self):
        return self._regulon_dict

    @regulon_dict.setter
    def regulon_dict(self, value):
        self._regulon_dict = value

    @property
    def auc_mtx(self):
        return self._auc_mtx

    @auc_mtx.setter
    def auc_mtx(self, value):
        self._auc_mtx = value

    # dotplot method for anndata
    @staticmethod
    def dotplot_anndata(data: anndata.AnnData,
                        gene_names: list,
                        cluster_label: str,
                        save: bool = True,
                        **kwargs):
        """
        create a dotplot for Anndata object.
        a dotplot contains percent (of cells that) expressed (the genes) and average expression (of genes).

        :param data: gene data
        :param gene_names: interested gene names
        :param cluster_label: label of clustering output
        :param save: if save plot into a file
        :param kwargs: features Input vector of features, or named list of feature vectors
        if feature-grouped panels are desired
        :return: plt axe object
        """
        if isinstance(data, anndata.AnnData):
            return sc.pl.dotplot(data, var_names=gene_names, groupby=cluster_label, save=save, **kwargs)
        elif isinstance(data, StereoExpData):
            logger.info('for StereoExpData object, please use function: dotplot_stereo')

    # dotplot method for StereoExpData
    @staticmethod
    def _cal_percent_df(exp_matrix: pd.DataFrame,
                        cluster_meta: pd.DataFrame,
                        regulon_genes: str,
                        celltype: list,
                        groupby: str,
                        cutoff: float = 0):
        """
        Expression percent
        cell numbers
        :param exp_matrix:
        :param cluster_meta:
        :param regulon_genes:
        :param celltype:
        :param cutoff:
        :return:
        """
        # which cells are in cluster X
        cells = cluster_meta[cluster_meta[groupby] == celltype]['cell']
        ncells = set(exp_matrix.index).intersection(set(cells))
        # get expression data for cells
        ct_exp = exp_matrix.loc[ncells]
        # input genes in regulon Y
        # get expression data for regulon Y genes in cluster X cells
        g_ct_exp = ct_exp[regulon_genes]
        # count the number of genes which expressed in cluster X cells
        regulon_cell_num = g_ct_exp[g_ct_exp > cutoff].count().count()
        total_cell_num = g_ct_exp.shape[0] * g_ct_exp.shape[1]
        if total_cell_num == 0:
            return 0
        else:
            return regulon_cell_num / total_cell_num

    @staticmethod
    def _cal_exp_df(exp_matrix, cluster_meta, regulon_genes, celltype: str, groupby: str):
        """
        Calculate average expression level for regulon Y genes in cluster X cells
        :param exp_matrix:
        :param cluster_meta:
        :param regulon_genes:
        :param celltype
        :return: numpy.float32
        """
        # get expression data for regulon Y genes in cluster X cells
        cells = cluster_meta[cluster_meta[groupby] == celltype]['cell']
        ncells = set(exp_matrix.index).intersection(set(cells))
        ct_exp = exp_matrix.loc[ncells]
        g_ct_exp = ct_exp[regulon_genes]
        if g_ct_exp.empty:
            return 0
        else:
            return np.mean(g_ct_exp)

    @staticmethod
    def dotplot_stereo(data: StereoExpData,
                       meta: pd.DataFrame,
                       regulon_dict,
                       regulon_names: list,
                       celltypes: list,
                       groupby: str,
                       palette: str = 'RdYlBu_r',
                       **kwargs):
        """
        Intuitive way of visualizing how feature expression changes across different
        identity classes (clusters). The size of the dot encodes the percentage of
        cells within a class, while the color encodes the AverageExpression level
        across all cells within a class (blue is high).

        :param celltypes:
        :param palette:
        :param groupby:
        :param regulon_names:
        :param regulon_dict:
        :param meta:
        :param data:
        :param kwargs: features Input vector of features, or named list of feature vectors
        if feature-grouped panels are desired
        :return:
        """
        expr_matrix = data.to_df()
        dot_data = {'cell type': [], 'regulons': [], 'percentage': [], 'avg exp': []}

        for reg in regulon_names:
            target_genes = regulon_dict[f'{reg}(+)']
            for ct in celltypes:
                reg_ct_percent = PlotRegulatoryNetwork._cal_percent_df(exp_matrix=expr_matrix,
                                                                       cluster_meta=meta,
                                                                       regulon_genes=target_genes,
                                                                       celltype=ct, groupby=groupby)
                reg_ct_avg_exp = PlotRegulatoryNetwork._cal_exp_df(exp_matrix=expr_matrix,
                                                                   cluster_meta=meta,
                                                                   regulon_genes=target_genes,
                                                                   celltype=ct, groupby=groupby)
                dot_data['regulons'].append(reg)
                dot_data['cell type'].append(ct)
                dot_data['percentage'].append(reg_ct_percent)
                dot_data['avg exp'].append(reg_ct_avg_exp)

        dot_df = pd.DataFrame(dot_data)
        dot_df.to_csv('dot_df.csv', index=False)
        g = sns.scatterplot(data=dot_df, size='percentage', hue='avg exp', x='regulons', y='cell type', sizes=(20, 200),
                            marker='o', palette=palette, legend='full', **kwargs)
        plt.legend(frameon=False, loc=(1.04, 0))
        plt.tick_params(axis='both', length=0, labelsize=6)
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig('dot.png')
        return g

    @staticmethod
    def auc_heatmap(auc_mtx: pd.DataFrame, width=8, height=8, fn='auc_heatmap.png', **kwargs):
        """
        Plot heatmap for auc value for regulons
        :param height:
        :param width:
        :param auc_mtx:
        :param fn:
        :return:
        """
        if 'Cell' in auc_mtx.columns:
            auc_mtx = auc_mtx.set_index('Cell')
        plt.figsize = (width, height)
        sns.clustermap(auc_mtx, **kwargs)
        plt.tight_layout()
        plt.savefig(fn)

    @staticmethod
    def plot_2d_reg_stereo(data: StereoExpData, auc_mtx, reg_name: str, **kwargs):
        """
        Plot genes of one regulon on a 2D map
        :param data:
        :param auc_mtx:
        :param reg_name:
        :return:
        """
        if '(+)' not in reg_name:
            reg_name = reg_name + '(+)'
        cell_coor = data.position
        auc_zscore = cal_zscore(auc_mtx)
        # prepare plotting data
        sub_zscore = auc_zscore[reg_name]
        # sort data points by zscore (low to high), because first dot will be covered by latter dots
        zorder = np.argsort(sub_zscore.values)
        # plot cell/bin dot, x y coor
        sc = plt.scatter(cell_coor[:, 0][zorder], cell_coor[:, 1][zorder], c=sub_zscore.iloc[zorder], marker='.',
                         edgecolors='none', cmap='plasma', lw=0, **kwargs)
        plt.box(False)
        plt.axis('off')
        plt.colorbar(sc, shrink=0.35)
        plt.savefig(f'{reg_name.split("(")[0]}.png')
        plt.close()

    @staticmethod
    def plot_2d_reg_h5ad(data: anndata.AnnData, pos_label, auc_mtx, reg_name: str, **kwargs):
        """
        Plot genes of one regulon on a 2D map
        :param pos_label:
        :param data:
        :param auc_mtx:
        :param reg_name:
        :return:

        Example:
            plot_2d_reg_h5ad(data, 'spatial', auc_mtx, 'Zfp354c')
        """
        if '(+)' not in reg_name:
            reg_name = reg_name + '(+)'
        cell_coor = data.obsm[pos_label]
        auc_zscore = cal_zscore(auc_mtx)
        # prepare plotting data
        sub_zscore = auc_zscore[reg_name]
        # sort data points by zscore (low to high), because first dot will be covered by latter dots
        zorder = np.argsort(sub_zscore.values)
        # plot cell/bin dot, x y coor
        sc = plt.scatter(cell_coor[:, 0][zorder], cell_coor[:, 1][zorder], c=sub_zscore.iloc[zorder], marker='.',
                         edgecolors='none', cmap='plasma', lw=0, **kwargs)
        plt.box(False)
        plt.axis('off')
        plt.colorbar(sc, shrink=0.35)
        plt.savefig(f'{reg_name.split("(")[0]}.png')
        plt.close()

    @staticmethod
    def rss_heatmap(data: anndata.AnnData,
                    auc_mtx: pd.DataFrame,
                    meta: pd.DataFrame,
                    regulons: list,
                    save=True,
                    fn='clusters-heatmap-top5.png',
                    **kwargs):
        """
        Plot heatmap for Regulon specificity scores (RSS) value
        :param data: 
        :param auc_mtx: 
        :param regulons:
        :param meta:
        :param save:
        :param fn:
        :return: 
        """
        meta = pd.read_csv('meta_mousebrain.csv', index_col=0).iloc[:, 0]

        # TODO: adapt to StereoExpData
        # load the regulon_list from a file using the load_signatures function
        # regulons = load_signatures(regulons_fn)  # regulons_df -> list of regulon_list
        # data = add_scenic_metadata(data, auc_mtx, regulons)

        # Regulon specificity scores (RSS) across predicted cell types
        rss_cellType = regulon_specificity_scores(auc_mtx, meta)
        rss_cellType.to_csv('regulon_specificity_scores.txt')
        # Select the top 5 regulon_list from each cell type
        cats = sorted(list(set(meta)))
        topreg = []
        for i, c in enumerate(cats):
            topreg.extend(
                list(rss_cellType.T[c].sort_values(ascending=False)[:5].index)
            )
        topreg = list(set(topreg))

        # plot z-score
        auc_zscore = cal_zscore(auc_mtx)
        sns.set(font_scale=1.2)
        g = sns.clustermap(auc_zscore[topreg], annot=False, square=False, linecolor='gray', yticklabels=True,
                           xticklabels=True, vmin=-2, vmax=6, cmap="YlGnBu", figsize=(21, 16), **kwargs)
        g.cax.set_visible(True)
        g.ax_heatmap.set_ylabel('')
        g.ax_heatmap.set_xlabel('')
        if save:
            plt.savefig(fn)
        return g


def cal_zscore(auc_mtx: pd.DataFrame) -> pd.DataFrame:
    """
    calculate z-score for each gene among cells
    :param auc_mtx:
    :return:
    """
    func = lambda x: (x - x.mean()) / x.std(ddof=0)
    auc_zscore = auc_mtx.transform(func, axis=0)
    auc_zscore.to_csv('auc_zscore.csv', index=False)
    return auc_zscore


def is_regulon(reg):
    """
    Decide if a string is a regulon_list name
    :param reg: the name of the regulon
    :return:
    """
    if '(+)' in reg or '(-)' in reg:
        return True
