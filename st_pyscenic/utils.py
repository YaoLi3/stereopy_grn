# -*- coding: utf-8 -*-

from itertools import chain
from typing import Sequence, Type

import pandas as pd
import numpy as np
from ctxcore.genesig import Regulon

from stereo.core.stereo_exp_data import StereoExpData
from stereo.log_manager import logger

import logging

LOGGER = logging.getLogger(__name__)

COLUMN_NAME_TF = "TF"
#COLUMN_NAME_MOTIF_ID = "MotifID"
#COLUMN_NAME_MOTIF_SIMILARITY_QVALUE = "MotifSimilarityQvalue"
#COLUMN_NAME_ORTHOLOGOUS_IDENTITY = "OrthologousIdentity"
#COLUMN_NAME_ANNOTATION = "Annotation"

COLUMN_NAME_TARGET = "target"
COLUMN_NAME_WEIGHT = "importance"
COLUMN_NAME_REGULATION = "regulation"
COLUMN_NAME_CORRELATION = "rho"
RHO_THRESHOLD = 0.03

ACTIVATING_MODULE = "activating"
REPRESSING_MODULE = "repressing"

def add_correlation(
    data: StereoExpData,
    adjacencies: pd.DataFrame,
    ex_mtx: pd.DataFrame,
    rho_threshold=RHO_THRESHOLD,
    mask_dropouts=False,
) -> pd.DataFrame:
    """
    Add correlation in expression levels between target and factor.
    :param adjacencies: The dataframe with the TF-target links.
    :param ex_mtx: The expression matrix (n_cells x n_genes).
    :param rho_threshold: The threshold on the correlation to decide if a target gene is activated
        (rho > `rho_threshold`) or repressed (rho < -`rho_threshold`).
    :param mask_dropouts: Do not use cells in which either the expression of the TF or the target gene is 0 when
        calculating the correlation between a TF-target pair.
    :return: The adjacencies dataframe with an extra column.
    """
    assert rho_threshold > 0, "rho_threshold should be greater than 0."

    # Assessment of best optimization strategy for calculating dropout masked correlations between TF-target expression:
    #
    # Measurement of time performance of masked_rho (with numba JIT): 136 µs ± 932 ns for a single pair of vectors.
    # For a typical dataset this translates into (for a single core):
    # 1. Calculating the rectangular (TFxtarget) correlation matrix:
    #    (1,564 TFs * 19,812 targets * 136 microseconds * 10e-6)/3600.0 ~ 12 hours.
    #    This approach calculates far too much be has the potential for easy parallelization via numba (cf. current
    #    implementation of masked_rho_2d).
    # 2. Calculating only needed TF-target pairs:
    #    (6,732,441 TF-target links * 136 microseconds * 10e-6)/3600.0 ~ 2h 30 mins.
    #    - Many of these gene-gene links will be duplicate so there might be a potential for memoization. However because
    #    the calculation is already quite fast and the memoization would need to take into account the commutativity of
    #    the operation and involves hashing large numerical vectors, the benefit if this memoization might be minimal.
    #    - Calculation of unique pairs already takes substantial amount of time and does not introduce a substantial
    #    reduction in the number of gene-gene pairs to calculate the correlation for: 6,732,441 => 6,630,720 (2 min 9 s).
    #    This is exactly the additional needed for calculating the rho values for these pairs. No gain here.
    #
    # The other options would have been to used the masked array abstraction provided by numpy but this again
    # this not allow for easy parallelization. In addition the corrcoef operation is far slower than the numba
    # JIT implementation: 2.36 ms ± 62 µs per loop.
    #
    # The best combined approach is to calculate rhos for pairs defined by indexes which is the approach implemented
    # below.

    # Calculate Pearson correlation to infer repression or activation.
    if mask_dropouts:
        ex_mtx = ex_mtx.sort_index(axis=1)
        col_idx_pairs = _create_idx_pairs(adjacencies, ex_mtx)
        rhos = masked_rho4pairs(ex_mtx.values, col_idx_pairs, 0.0)
    else:
        genes = list(
            set(adjacencies[COLUMN_NAME_TF]).union(set(adjacencies[COLUMN_NAME_TARGET]))
        )
        # TODO: how to subset StereoExpData matrix??
        #ex_mtx = ex_mtx[ex_mtx.columns[ex_mtx.columns.isin(genes)]]
        corr_mtx = pd.DataFrame(
            index=data.gene_names,
            columns=data.gene_names,
            data=np.corrcoef(data.exp_matrix.T),  # TODO: why transpose??
        )
        rhos = np.array(
            [corr_mtx[s2][s1] for s1, s2 in zip(adjacencies.TF, adjacencies.target)]
        )

    regulations = (rhos > rho_threshold).astype(int) - (rhos < -rho_threshold).astype(
        int
    )
    return pd.DataFrame(
        data={
            COLUMN_NAME_TF: adjacencies[COLUMN_NAME_TF].values,
            COLUMN_NAME_TARGET: adjacencies[COLUMN_NAME_TARGET].values,
            COLUMN_NAME_WEIGHT: adjacencies[COLUMN_NAME_WEIGHT].values,
            COLUMN_NAME_REGULATION: regulations,
            COLUMN_NAME_CORRELATION: rhos,
        }
    )


def modules_from_adjacencies(
    data: StereoExpData,
    adjacencies: pd.DataFrame,
    ex_mtx: pd.DataFrame,
    thresholds=(0.75, 0.90),
    top_n_targets=(50,),
    top_n_regulators=(5, 10, 50),
    min_genes=20,
    absolute_thresholds=False,
    rho_dichotomize=True,
    keep_only_activating=True,
    rho_threshold=RHO_THRESHOLD,
    rho_mask_dropouts=False,
) -> Sequence[Regulon]:
    """
    Create modules from a dataframe containing weighted adjacencies between a TF and its target genes.
    :param adjacencies: The dataframe with the TF-target links. This dataframe should have the following columns:
        :py:const:`pyscenic.utils.COLUMN_NAME_TF`, :py:const:`pyscenic.utils.COLUMN_NAME_TARGET` and :py:const:`pyscenic.utils.COLUMN_NAME_WEIGHT` .
    :param ex_mtx: The expression matrix (n_cells x n_genes).
    :param thresholds: the first method to create the TF-modules based on the best targets for each transcription factor.
    :param top_n_targets: the second method is to select the top targets for a given TF.
    :param top_n_regulators: the alternative way to create the TF-modules is to select the best regulators for each gene.
    :param min_genes: The required minimum number of genes in a resulting module.
    :param absolute_thresholds: Use absolute thresholds or percentiles to define modules based on best targets of a TF.
    :param rho_dichotomize: Differentiate between activating and repressing modules based on the correlation patterns of
        the expression of the TF and its target genes.
    :param keep_only_activating: Keep only modules in which a TF activates its target genes.
    :param rho_threshold: The threshold on the correlation to decide if a target gene is activated
        (rho > `rho_threshold`) or repressed (rho < -`rho_threshold`).
    :param rho_mask_dropouts: Do not use cells in which either the expression of the TF or the target gene is 0 when
        calculating the correlation between a TF-target pair.
    :return: A sequence of regulons.
    """

    # Duplicate genes need to be removed from the expression matrix to avoid lookup problems in the correlation
    # matrix.
    # In addition, also make sure the expression matrix consists of floating point numbers. This requirement might
    # be violated when dealing with raw counts as input.
    
    # TODO: !!! remove duplicate genes in StereoExpData !!!
    #ex_mtx = ex_mtx.T[~ex_mtx.columns.duplicated(keep="first")].T.astype(float)

    # To make the pySCENIC code more robust to the selection of the network inference method in the first step of
    # the pipeline, it is better to use percentiles instead of absolute values for the weight thresholds.
    if not absolute_thresholds:

        def iter_modules(adjc, context):
            yield from chain(
                chain.from_iterable(
                    modules4thr(
                        adjc, thr, context, pattern="weight>{}%".format(frac * 100)
                    )
                    for thr, frac in zip(
                        list(adjacencies[COLUMN_NAME_WEIGHT].quantile(thresholds)),
                        thresholds,
                    )
                ),
                chain.from_iterable(
                    modules4top_targets(adjc, n, context) for n in top_n_targets
                ),
                chain.from_iterable(
                    modules4top_factors(adjc, n, context) for n in top_n_regulators
                ),
            )

    else:

        def iter_modules(adjc, context):
            yield from chain(
                chain.from_iterable(
                    modules4thr(adjc, thr, context) for thr in thresholds
                ),
                chain.from_iterable(
                    modules4top_targets(adjc, n, context) for n in top_n_targets
                ),
                chain.from_iterable(
                    modules4top_factors(adjc, n, context) for n in top_n_regulators
                ),
            )

    if not rho_dichotomize:
        # Do not differentiate between activating and repressing modules.
        modules_iter = iter_modules(adjacencies, frozenset())
    else:
        # Relationship between TF and its target, i.e. activator or repressor, is derived using the original expression
        # profiles. The Pearson product-moment correlation coefficient is used to derive this information.

        if not {"regulation", "rho"}.issubset(adjacencies.columns):
            # Add correlation column and create two disjoint set of adjacencies.
            print("Calculating Pearson correlations.")
            # test for genes present in the adjacencies but not present in the expression matrix:
            unique_adj_genes = set(adjacencies[COLUMN_NAME_TF]).union(
                set(adjacencies[COLUMN_NAME_TARGET])
            ) - set(data.gene_names)
            assert (
                len(unique_adj_genes) == 0
            ), f"Found {len(unique_adj_genes)} genes present in the network (adjacencies) output, but missing from the expression matrix. Is this a different gene expression matrix?"
            LOGGER.warn(
                f"Note on correlation calculation: the default behaviour for calculating the correlations has changed after pySCENIC verion 0.9.16. Previously, the default was to calculate the correlation between a TF and target gene using only cells with non-zero expression values (mask_dropouts=True). The current default is now to use all cells to match the behavior of the R verision of SCENIC. The original settings can be retained by setting 'rho_mask_dropouts=True' in the modules_from_adjacencies function, or '--mask_dropouts' from the CLI.\n\tDropout masking is currently set to [{rho_mask_dropouts}]."
            )
            adjacencies = add_correlation(
                data,
                adjacencies,
                ex_mtx,
                rho_threshold=rho_threshold,
                mask_dropouts=rho_mask_dropouts,
            )
        else:
            print(
                "Using existing Pearson correlations from the adjacencies file."
            )

        activating_modules = adjacencies[adjacencies[COLUMN_NAME_REGULATION] > 0.0]
        if keep_only_activating:
            modules_iter = iter_modules(
                activating_modules, frozenset([ACTIVATING_MODULE])
            )
        else:
            repressing_modules = adjacencies[adjacencies[COLUMN_NAME_REGULATION] < 0.0]
            modules_iter = chain(
                iter_modules(activating_modules, frozenset([ACTIVATING_MODULE])),
                iter_modules(repressing_modules, frozenset([REPRESSING_MODULE])),
            )

    # Derive modules for these adjacencies.
    # + Add the transcription factor to the module.
    #   [We are unable to assess if a TF works in a direct self-regulating way, either inhibiting its own expression or
    #    activating it. Therefore the most unbiased way forward is to add the TF to both activating as well as
    #    repressing modules]
    # + Filter for minimum number of genes.
    print("Creating modules.")

    def add_tf(module):
        return module.add(module.transcription_factor)

    return list(filter(lambda m: len(m) >= min_genes, map(add_tf, modules_iter)))


