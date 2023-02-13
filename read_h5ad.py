import stereo as st
import scanpy as sc
from scanpy.pl import dotplot 
import matplotlib.pyplot as plt


fn = '/dellfsqd2/ST_OCEAN/USER/liyao1/stereopy/resource/StereopyData/MouseBrainCellbin.h5ad'

data = sc.read_h5ad(fn)
print(data.var_names)

genes = ['0610005C13Rik', '0610009B22Rik', '0610009O20Rik', '0610010F05Rik','0610010K14Rik', '0610012G03Rik', '0610025J13Rik', '0610030E20Rik','0610033M10Rik', '0610037L13Rik']

dotplot(data, var_names=genes, groupby='psuedo_class', save=True)

