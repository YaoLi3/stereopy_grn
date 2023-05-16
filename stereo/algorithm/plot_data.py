import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import sys
import os
import matplotlib.patches as mpatches
import glob
import numpy as np


def highlight_key(color_dir: dict, new_value: str='#8a8787', key_to_highlight: list=['Cardiac muscle lineages']):
    """
    Highlight one or more interested keys/celltypes when plotting,
    the rest of keys/celltypes will be set to gray by default.
    """
    #assert key_to_highlight in color_dir.keys()
    for k, v in color_dir.items():
        if k not in key_to_highlight:
            color_dir[k] = new_value
    return color_dir


def plot_legend(color_dir, name=''):
    fig = plt.figure(figsize=(10, 5))
    markers = [plt.Line2D([0,0],[0,0], color=color, marker='o', linestyle='') for color in color_dir.values()]
    plt.legend(markers, color_dir.keys(), numpoints=1, ncol=3, loc='center')
    plt.box(False)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f'{name}_legend.png')
    plt.close()


def plot_2d(adata, color_dir, pos_label='spatial_regis', cluster_label='ctype_user', name='test'):
    celltypes = list(adata.obs[cluster_label])  # cell order = origional order in the array
    color_map = [color_dir[c] for c in celltypes]
    plt.scatter(x=adata.obsm[pos_label][:,0], y=adata.obsm[pos_label][:,1], c=color_map, marker='.', s=0.15)
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f'{name}.png')
    plt.close()


if __name__ == '__main__':
    ## PLOT 2D
    #adata = sc.read_h5ad(sys.argv[1])
    #name = os.path.basename(sys.argv[1]).split('.')[0]
    # important: 自定义celltype color
    #celltypes = list(adata.obs['ctype_user'])  # CELL ORDER!
    #color_dir = adata.uns['color_ManualAnnotation']
    #color_map = [color_dir[c] for c in celltypes]
    #color_dir_gray = highlight_key(color_dir) 
    #color_map_gray = [color_dir_gray[c] for c in celltypes]
    #plt.scatter(x=adata.obsm['spatial_regis'][:,0], y=adata.obsm['spatial_regis'][:,1], c=color_map, marker='.', s=0.15)
    #plt.axis("equal")
    #plt.tight_layout()
    #plt.savefig(f'{name}.png')
    #plt.close()

    #plt.scatter(x=adata.obsm['spatial_regis'][:,0], y=adata.obsm['spatial_regis'][:,1], c=color_map_gray, marker='.', s=0.15)
    #plt.axis("equal")
    #plt.tight_layout()
    #plt.savefig(f'{name}_gray.png')
    #plt.close()

    ## PLOT 3D
    # sort file by slice ID
    file_list = map(os.path.basename, glob.glob('./*.h5ad'))
    file_list = sorted(file_list, key=lambda x: int(x.split('_')[0]))
    # subset slices if needed
    file_list = list(filter(lambda x: int(x.split('_')[0]) >= 8 and int(x.split('_')[0]) <= 65, file_list))

    # concatenate all data into one
    total_cell_coor = []
    total_color_map = []
    total_color_map_gray = []
    for fn in file_list:
        adata = sc.read_h5ad(fn)
        # extract cell coordinates
        cell_coor = adata.obsm['spatial_regis']
        celltypes = list(adata.obs['ctype_user'])  # CELL ORDER!
        # extract celltype colors
        color_dir = adata.uns['color_ManualAnnotation']
        color_map = [color_dir[c] for c in celltypes]
        color_dir_gray = highlight_key(color_dir) 
        color_map_gray = [color_dir_gray[c] for c in celltypes]
        
        # save data from multiple slices into one variable
        total_cell_coor.append(cell_coor)
        total_color_map.extend(color_map)
        total_color_map_gray.extend(color_map_gray)
    # concat cell coor arrays
    total_cell_coor = np.ma.concatenate(total_cell_coor)
   
    # set angle value
    vview = 60
    hview = 0
    # plotting 
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    scp = ax.scatter(total_cell_coor[:, 0],
                    total_cell_coor[:, 1],
                    total_cell_coor[:, 2],
                    c=total_color_map,
                    marker='.',
                    edgecolors='none',
                    lw=0,
                    s=0.15)
    # set view angle
    ax.view_init(vview, hview)
    # scale axis
    xlen = total_cell_coor[:, 0].max() - total_cell_coor[:, 0].min()
    ylen = total_cell_coor[:, 1].max() - total_cell_coor[:, 1].min()
    zlen = total_cell_coor[:, 2].max() - total_cell_coor[:, 2].min()
    yscale = ylen / xlen
    zscale = zlen / xlen
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, yscale, zscale, 1]))

    plt.box(False)
    plt.axis('off')
    plt.colorbar(scp, shrink=0.35)
    plt.savefig('mouse_embryo_3D.png')
    plt.close()


    fig = plt.figure()
    ax = Axes3D(fig)
    scp = ax.scatter(total_cell_coor[:, 0],
                    total_cell_coor[:, 1],
                    total_cell_coor[:, 2],
                    c=total_color_map_gray,
                    marker='.',
                    edgecolors='none',
                    lw=0,
                    s=0.15)
    # set view angle
    ax.view_init(vview, hview)
    # scale axis
    xlen = total_cell_coor[:, 0].max() - total_cell_coor[:, 0].min()
    ylen = total_cell_coor[:, 1].max() - total_cell_coor[:, 1].min()
    zlen = total_cell_coor[:, 2].max() - total_cell_coor[:, 2].min()
    yscale = ylen / xlen
    zscale = zlen / xlen
    ax.get_proj = lambda: np.dot(Axes3D.get_proj(ax), np.diag([1, yscale, zscale, 1]))

    plt.box(False)
    plt.axis('off')
    plt.colorbar(scp, shrink=0.35)
    plt.savefig('mouse_embryo_3D_gray.png')
    plt.close()




