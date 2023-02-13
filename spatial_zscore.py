
import anndata
import pandas as pd

coords = pd.read_csv('/dellfsqd2/ST_OCEAN/USER/hankai/Project/10.smed/11.web_data/smed.dd_g4.NsC/NsC_seurat_meta.txt', 
        sep='\t', header=0, index_col=0)
coords = coords[['timepoint']]
zscore = pd.read_csv('/dellfsqd2/ST_OCEAN/USER/hankai/Project/10.smed/26.SCENIC/Regeneration.med8/pyscenic_output.prune.zscore.csv', 
        sep=',', header=0, index_col=0)

zscore = zscore.merge(coords, how='left', left_index=True, right_index=True)
zscore = zscore.reset_index()
zscore[['index', 'b']] = zscore['index'].str.rsplit('_', n=1, expand=True)
zscore['index'] = zscore['index'] + '.' + zscore['timepoint']
del zscore['timepoint']
del zscore['b']
zscore = zscore.set_index('index')

import sys
sys.path.append('/dellfsqd2/ST_OCEAN/USER/hankai/Project/10.smed/code/site-packages')
from WBRtools.coords3d import Coords
from WBRtools.mipplot import mip_spatial

order = ['0hpa1', '12hpa2', '36hpa2', '3dpa2', '5dpa1', '7dpa2', '10dpa1', '14dpa1']
adata = anndata.read('/dellfsqd2/ST_OCEAN/USER/hankai/Project/10.smed/11.web_data/smed.dd_g4.NsC/NsC.h5ad')
adata.obs = adata.obs.merge(zscore, how='left', left_index=True, right_index=True)
adata = adata[adata.obs['timepoint'].isin(order)]
adata.uns['3d_spatial'] = adata.obs[['x', 'y', 'z']].values

#coords = Coords(coords)
for col in zscore.columns:
    if '(+)' not in col:
        continue
    if col != 'SMESG000045451.1(+)':
        continue
    
    mip_spatial(adata, 
        color=col,
        group_col='timepoint', groups=order,
        outfile=f'./zscore/{col}.pdf', 
        alpha_shape=False, alpha=0.07,
        theme='dark', as_channels=False, 
        cmap=None,
        body_loc='selectted'
        )




