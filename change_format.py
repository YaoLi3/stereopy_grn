# 2022-DEC
# YAO

'''
convert 10X visum files to one loom file
or 
convert csv to loom
'''

import os, sys
os.getcwd()
os.listdir(os.getcwd()) 

import loompy as lp
import numpy as np
import scanpy as sc

dirn = sys.argv[1]
adata = sc.read_10x_mtx(
    dirn,                 # the directory with the `.mtx` file
    var_names='gene_symbols',   # use gene symbols for the variable names (variables-axis index)
    cache=True) 
print(adata.var)
print(adata.obs)
print(adata.X)

row_attrs = { 
    "Gene": np.array(adata.var.index) ,
}

col_attrs = { 
    "CellID":  np.array(adata.obs.index) ,
    "nGene": np.array( np.sum(adata.X.transpose()>0 , axis=0)).flatten() ,
    "nUMI": np.array( np.sum(adata.X.transpose() , axis=0)).flatten() ,
}

lp.create('unfiltered.loom', adata.X.transpose(), row_attrs, col_attrs )


#fn = os.path.basename(sys.argv[1])
#x=sc.read_csv(fn);
#row_attrs = {"Gene": np.array(x.var_names),};
#col_attrs = {"CellID": np.array(x.obs_names)};
#lp.create(f"{fn.split('.')[0]}.loom",x.X.transpose(),row_attrs,col_attrs);


